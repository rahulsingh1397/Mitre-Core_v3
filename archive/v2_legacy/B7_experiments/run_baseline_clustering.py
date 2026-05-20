"""
experiments/run_baseline_clustering.py
--------------------------------------
Baseline clustering comparison for MITRE-CORE publication readiness.
Compares K-means, DBSCAN, HDBSCAN, Spectral Clustering vs MITRE-CORE GAEC.

Usage:
    python experiments/run_baseline_clustering.py \
        --output experiments/results/baseline_clustering_comparison.csv

Datasets: UNSW-NB15, NSL-KDD, BETH, OpTC
Metrics: ARI, NMI, Silhouette Score
"""

import argparse
import time
import logging
import sys
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score, silhouette_score
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
import torch
import logging
import gc
import time
from typing import Optional, List

from sklearn.decomposition import PCA
import hdbscan
# Add root and current directory to path BEFORE importing local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hgnn.hgnn_correlation import HGNNCorrelationEngine
from run_gate_tuning import DATASET_CONFIG as GATE_TUNING_CONFIG
# HGNN embedding methods are now enabled

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
from utils.seed_control import set_seed

# -----------------------------------------------------------------------
# Dataset configuration: merge gate-tuning paths/labels with baseline-specific parameters
# -----------------------------------------------------------------------

BASELINE_DATASETS = {
    "UNSW-NB15": {
        "true_clusters": 8,
        "hdbscan_min_cluster_size": 15,
        "dbscan_eps": 0.5,
    },
    "NSL-KDD": {
        "true_clusters": 10,  # tactic labels (not campaign_id)
        "hdbscan_min_cluster_size": 30,
        "dbscan_eps": 0.3,
    },
    "CICIDS2017": {
        "true_clusters": 16,  # campaign_id from mitre_format.parquet
        "hdbscan_min_cluster_size": 20,
        "dbscan_eps": 0.4,
    },
    "TON_IoT": {
        "true_clusters": 10,  # campaign_id from mitre_format.parquet
        "hdbscan_min_cluster_size": 15,
        "dbscan_eps": 0.3,
    },
    "SQTK_SIEM_kcluster": {
        "true_clusters": 11,  # kcluster from mitre_core_format.csv
        "hdbscan_min_cluster_size": 5,
        "dbscan_eps": 0.3,
    },
    "OpTC": {
        "true_clusters": 2,   # binary classification (RedTeam vs Benign)
        "hdbscan_min_cluster_size": 50,
        "dbscan_eps": 0.6,
    }
}

# Merge path/label info from gate tuning config to ensure consistency
for name, extras in BASELINE_DATASETS.items():
    if name in GATE_TUNING_CONFIG:
        extras.update({k: v for k, v in GATE_TUNING_CONFIG[name].items() 
                      if k in ("path", "label_col", "sample_size", "stratified_sample")})
    else:
        logger.warning(f"Dataset {name} not found in GATE_TUNING_CONFIG")

DATASET_CONFIG = BASELINE_DATASETS

def _make_baseline_methods(seed: int = 42) -> dict:
    """Factory that returns baseline methods parameterised by seed."""
    return {
        "K-Means": lambda n_clusters: KMeans(n_clusters=n_clusters, random_state=seed, n_init=10),
        "DBSCAN": lambda eps: DBSCAN(eps=eps, min_samples=5),
        "Spectral": lambda n_clusters: SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity='nearest_neighbors'),
        "HDBSCAN": lambda min_cluster_size: hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    }

BASELINE_METHODS = _make_baseline_methods(42)

def extract_hgnn_embeddings(df, checkpoint_path, config):
    """Extract HGNN backbone embeddings and instantiate engine."""
    try:
        engine = HGNNCorrelationEngine(
            model_path=checkpoint_path, 
            device="cpu", 
            use_geometric_confidence=True,
            pure_unsupervised=True,
            hdbscan_min_cluster_size=config.get("hdbscan_min_cluster_size", 15),
            hdbscan_pca_components=config.get("hdbscan_pca_components", 16),
            hdbscan_cluster_selection_epsilon=config.get("hdbscan_cluster_selection_epsilon", 0.1),
            hdbscan_use_umap=config.get("use_umap", False),
            hdbscan_umap_n_components=config.get("umap_n_components", 10),
        )
        return engine.extract_embeddings(df), engine
    except Exception as e:
        logger.error(f"Failed to extract HGNN embeddings: {e}")
        return None, None

def load_and_preprocess_dataset(dataset_name: str, config: dict) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Load dataset and return (df, features, n_true_classes)."""
    logger.info(f"Loading {dataset_name} dataset...")
    
    # Load the dataset
    data_path = Path(config["path"])
    if not data_path.exists():
        logger.warning(f"Dataset {dataset_name} not found at {data_path}")
        return None, None, None
    
    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    
    # Add sample size cap to prevent OOM (same as gate sweep)
    SAMPLE_SIZE = 10000
    if len(df) > SAMPLE_SIZE:
        logger.info(f"Sampling {SAMPLE_SIZE} records from {len(df)} to prevent OOM")
        df = df.sample(SAMPLE_SIZE, random_state=config.get("_seed", 42)).reset_index(drop=True)
    
    # Add debug logging to check dataset loading
    logger.info(f"Loaded {dataset_name}: {len(df)} samples, {len(df.columns)} features")
    logger.info(f"Available columns: {list(df.columns)}")
    
    # Dynamic feature selection: use all numeric columns except excluded ones
    exclude_cols = {config["label_col"], "AlertId", "EndDate", "MalwareIntelAttackType",
                    "AttackTechnique", "correlation_method", "pred_cluster"}
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude_cols]
    available_cols = numeric_cols
    
    if len(available_cols) == 0:
        logger.warning(f"No numeric feature columns available for {dataset_name}")
        return None, None, None
    
    logger.info(f"Using {len(available_cols)} numeric features: {available_cols[:10]}{'...' if len(available_cols) > 10 else ''}")
    
    # Extract features and labels
    features_df = df[available_cols].copy()
    labels = LabelEncoder().fit_transform(df[config["label_col"]].fillna("UNKNOWN").astype(str))
    n_true_classes = config["true_clusters"]
    
    # Handle categorical variables by converting to numeric
    categorical_cols = features_df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
    
    # Convert to numpy array and fill NaN values
    features = features_df.fillna(0).values
    
    # Add debug logging for feature analysis
    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")
    logger.info(f"Feature statistics - min: {features.min()}, max: {features.max()}, mean: {features.mean()}")
    
    # Check for all-zero features (common cause of zero ARI)
    zero_variance_cols = np.where(features.std(axis=0) == 0)[0]
    if len(zero_variance_cols) > 0:
        logger.warning(f"Zero variance features detected: {len(zero_variance_cols)} columns")
    
    # Check for NaN/inf values
    if np.isnan(features).any() or np.isinf(features).any():
        logger.warning(f"NaN or inf values detected in features")
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return df, features, n_true_classes

def evaluate_clustering(true_labels, pred_labels, features, df, label_col, n_true_classes):
    """Evaluate clustering performance with binary_ari for 2-class datasets."""
    # Handle noise points in DBSCAN/HDBSCAN (label = -1)
    mask = pred_labels != -1
    if mask.sum() == 0:
        return {"ari": 0.0, "nmi": 0.0, "silhouette": 0.0, "coverage": 0.0, "binary_ari": float("nan")}
    
    ari = adjusted_rand_score(true_labels[mask], pred_labels[mask])
    nmi = normalized_mutual_info_score(true_labels[mask], pred_labels[mask])
    ami = adjusted_mutual_info_score(true_labels[mask], pred_labels[mask])
    homogeneity = homogeneity_score(true_labels[mask], pred_labels[mask])
    completeness = completeness_score(true_labels[mask], pred_labels[mask])
    v_measure = v_measure_score(true_labels[mask], pred_labels[mask])
    
    # Silhouette score only if we have >1 cluster and >1 point per cluster, and reasonable dataset size
    unique_labels = np.unique(pred_labels[mask])
    mask_size = mask.sum()
    if len(unique_labels) > 1 and mask_size > len(unique_labels) and mask_size < 10000:
        try:
            silhouette = silhouette_score(features[mask], pred_labels[mask])
        except (MemoryError, ValueError):
            logger.warning(f"Silhouette calculation failed for {mask_size} samples, using 0.0")
            silhouette = 0.0
    else:
        if mask_size >= 10000:
            logger.warning(f"Skipping silhouette calculation for large dataset ({mask_size} samples)")
        silhouette = 0.0
    
    coverage = mask.mean()  # Fraction of points assigned to clusters
    
    # Binary ARI for 2-class datasets (same logic as run_gate_tuning.py)
    if n_true_classes == 2:
        # Create result DataFrame for binary ARI computation
        result_df = pd.DataFrame({
            'pred_cluster': pred_labels,
            label_col: df[label_col]
        })
        cluster_majority = (
            result_df.groupby("pred_cluster")[label_col]
            .agg(lambda x: x.mode().iloc[0])
        )
        binary_pred = result_df["pred_cluster"].map(cluster_majority)
        binary_ari = float(adjusted_rand_score(result_df[label_col], binary_pred))
    else:
        binary_ari = float("nan")
    
    return {
        "ari": ari,
        "nmi": nmi, 
        "ami": float(ami),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "v_measure": float(v_measure),
        "silhouette": silhouette,
        "coverage": coverage,
        "binary_ari": binary_ari
    }

def get_mitre_core_results(dataset_name):
    """Get MITRE-CORE results from v34 CSV instead of re-running."""
    try:
        v34_path = Path("experiments/results/gate_tuning_results_v34_final.csv")
        if not v34_path.exists():
            logger.warning(f"v34 results not found: {v34_path}")
            return None
        
        v34_df = pd.read_csv(v34_path)
        
        # Get best result per dataset (gate=0.5 for UNSW, 0.4 for others)
        dataset_results = v34_df[v34_df["dataset"] == dataset_name]
        
        if dataset_name == "UNSW-NB15":
            best_result = dataset_results[dataset_results["gate_value"] == 0.5].iloc[0]
        else:
            # For other datasets, use gate=0.4 (best for OpTC binary_ari)
            best_result = dataset_results[dataset_results["gate_value"] == 0.4].iloc[0]
        
        return {
            "ari": best_result["ari"],
            "binary_ari": best_result["binary_ari"],
            "nmi": best_result["nmi"],
            "silhouette": 0.0,  # Not in v34, set to 0
            "coverage": 1.0,   # HGNN covers all points
            "n_clusters": best_result["n_clusters"]
        }
        
    except Exception as e:
        logger.error(f"Failed to get MITRE-CORE results for {dataset_name}: {e}")
        return None

def run_baseline_comparison(seed: int = 42):
    """Run complete baseline clustering comparison with given seed."""
    set_seed(seed)
    baseline_methods = _make_baseline_methods(seed)
    results = []
    checkpoint_path = "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
    
    for dataset_name, config in DATASET_CONFIG.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        # Inject seed into config for sampling
        config["_seed"] = seed
        
        # Load data
        df, features, n_true_classes = load_and_preprocess_dataset(dataset_name, config)
        if df is None:
            continue
        
        # Extract labels for evaluation
        labels = LabelEncoder().fit_transform(df[config["label_col"]].fillna("UNKNOWN").astype(str))
        
        # Extract HGNN embeddings for embedding-based methods
        embeddings, engine = extract_hgnn_embeddings(df, checkpoint_path, config)
        if embeddings is None:
            logger.error("Failed to load HGNN embeddings. Exiting.")
            continue
        
        # Run baseline methods on raw features
        for method_name, method_factory in baseline_methods.items():
            logger.info(f"Running {method_name} on {dataset_name}...")
            
            try:
                if method_name == "K-Means":
                    model = method_factory(config["true_clusters"])
                elif method_name == "DBSCAN":
                    model = method_factory(config["dbscan_eps"])
                elif method_name == "Spectral":
                    model = method_factory(config["true_clusters"])
                elif method_name == "HDBSCAN":
                    model = method_factory(config["hdbscan_min_cluster_size"])
                
                start_time = time.time()
                pred_labels = model.fit_predict(features)
                runtime = time.time() - start_time
                
                metrics = evaluate_clustering(labels, pred_labels, features, df, config["label_col"], n_true_classes)
                
                result = {
                    "dataset": dataset_name,
                    "method": method_name,
                    "runtime_seconds": runtime,
                    "n_true_classes": n_true_classes,
                    **metrics
                }
                results.append(result)
                
                logger.info(f"{method_name}: ARI={metrics['ari']:.3f}, binary_ARI={metrics['binary_ari']:.3f}, Coverage={metrics['coverage']:.3f}")
                
            except Exception as e:
                logger.error(f"{method_name} failed on {dataset_name}: {e}")
                results.append({
                    "dataset": dataset_name,
                    "method": method_name,
                    "runtime_seconds": 0.0,
                    "n_true_classes": n_true_classes,
                    "ari": 0.0,
                    "binary_ari": float("nan"),
                    "nmi": 0.0,
                    "ami": float("nan"),
                    "homogeneity": float("nan"),
                    "completeness": float("nan"),
                    "v_measure": float("nan"),
                    "silhouette": 0.0,
                    "coverage": 0.0
                })
        
        # Run embedding-based methods if embeddings available
        if embeddings is not None:
            logger.info(f"Running embedding-based methods on {dataset_name}...")
            
            # PCA reduce (same as GAEC)
            pca = PCA(n_components=16, whiten=True, random_state=seed)
            z = pca.fit_transform(embeddings)

            # K-means on embeddings
            try:
                model = KMeans(n_clusters=config["true_clusters"], random_state=seed, n_init=20)
                start_time = time.time()
                pred_labels = model.fit_predict(z)
                runtime = time.time() - start_time
                
                metrics = evaluate_clustering(labels, pred_labels, embeddings, df, config["label_col"], n_true_classes)
                
                result = {
                    "dataset": dataset_name,
                    "method": "K-Means-emb",
                    "runtime_seconds": runtime,
                    "n_true_classes": n_true_classes,
                    **metrics
                }
                results.append(result)
                
                logger.info(f"K-Means-emb: ARI={metrics['ari']:.3f}, binary_ARI={metrics['binary_ari']:.3f}")
                
            except Exception as e:
                logger.error(f"K-Means-emb failed on {dataset_name}: {e}")
            
            # Spectral on embeddings
            try:
                model = SpectralClustering(n_clusters=config["true_clusters"], random_state=seed, affinity='nearest_neighbors')
                start_time = time.time()
                pred_labels = model.fit_predict(z)
                runtime = time.time() - start_time
                
                metrics = evaluate_clustering(labels, pred_labels, embeddings, df, config["label_col"], n_true_classes)
                
                result = {
                    "dataset": dataset_name,
                    "method": "Spectral-emb",
                    "runtime_seconds": runtime,
                    "n_true_classes": n_true_classes,
                    **metrics
                }
                results.append(result)
                
                logger.info(f"Spectral-emb: ARI={metrics['ari']:.3f}, binary_ARI={metrics['binary_ari']:.3f}")
                
            except Exception as e:
                logger.error(f"Spectral-emb failed on {dataset_name}: {e}")
            
            # HDBSCAN on embeddings
            try:
                model = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean')
                start_time = time.time()
                pred_labels = model.fit_predict(z)
                runtime = time.time() - start_time
                
                metrics = evaluate_clustering(labels, pred_labels, embeddings, df, config["label_col"], n_true_classes)
                
                result = {
                    "dataset": dataset_name,
                    "method": "HDBSCAN-emb",
                    "runtime_seconds": runtime,
                    "n_true_classes": n_true_classes,
                    **metrics
                }
                results.append(result)
                
                logger.info(f"HDBSCAN-emb: ARI={metrics['ari']:.3f}, binary_ARI={metrics['binary_ari']:.3f}")
                
            except Exception as e:
                logger.error(f"HDBSCAN-emb failed on {dataset_name}: {e}")

            # MITRE-CORE (existing engine.correlate())
            try:
                start_time = time.time()
                mitre_result = engine.correlate(df)
                pred_labels = mitre_result["pred_cluster"].values
                runtime = time.time() - start_time
                
                metrics = evaluate_clustering(labels, pred_labels, embeddings, df, config["label_col"], n_true_classes)
                
                result = {
                    "dataset": dataset_name,
                    "method": "MITRE-CORE",
                    "runtime_seconds": runtime,
                    "n_true_classes": n_true_classes,
                    **metrics
                }
                results.append(result)
                
                logger.info(f"MITRE-CORE: ARI={metrics['ari']:.3f}, binary_ARI={metrics['binary_ari']:.3f}")
                
            except Exception as e:
                logger.error(f"MITRE-CORE failed on {dataset_name}: {e}")
        else:
            logger.warning(f"Could not extract embeddings for {dataset_name}, skipping embedding-based methods")
        
        # Memory cleanup
        gc.collect()
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run baseline clustering comparison")
    parser.add_argument("--output", default="experiments/results/baseline_clustering_comparison.csv",
                       help="Output CSV file for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (single-seed mode)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                       help="Multiple seeds for multi-seed aggregation (overrides --seed)")
    args = parser.parse_args()
    
    seeds = args.seeds if args.seeds else [args.seed]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    for seed in seeds:
        logger.info(f"\n{'#'*80}")
        logger.info(f"Running baseline comparison with seed={seed}")
        logger.info(f"{'#'*80}")
        results = run_baseline_comparison(seed=seed)
        for r in results:
            r["seed"] = seed
        all_results.extend(results)
    
    df_results = pd.DataFrame(all_results)
    
    # Save per-seed raw results
    df_results.to_csv(output_path, index=False)
    logger.info(f"\nPer-seed results saved to: {output_path}")
    
    # If multi-seed, produce aggregated summary (mean ± std)
    if len(seeds) > 1:
        agg = df_results.groupby(["dataset", "method"]).agg(
            ari_mean=("ari", "mean"), ari_std=("ari", "std"),
            ami_mean=("ami", "mean"), ami_std=("ami", "std"),
            binary_ari_mean=("binary_ari", "mean"), binary_ari_std=("binary_ari", "std"),
            nmi_mean=("nmi", "mean"), nmi_std=("nmi", "std"),
        ).reset_index()
        summary_path = output_path.with_name(output_path.stem + "_summary" + output_path.suffix)
        agg.to_csv(summary_path, index=False)
        logger.info(f"Multi-seed summary saved to: {summary_path}")
        logger.info("\n" + agg.to_string(index=False))
    else:
        # Print summary table
        logger.info("\n" + "="*80)
        logger.info("BASELINE CLUSTERING COMPARISON SUMMARY")
        logger.info("="*80)
        summary_pivot = df_results.pivot(index='dataset', columns='method', values='ari')
        logger.info("\nARI Scores by Dataset and Method:")
        logger.info(summary_pivot.round(4).to_string())
    
    # Find best method per dataset
    best_methods = df_results.loc[df_results.groupby('dataset')['ari'].idxmax()]
    logger.info("\nBest Method per Dataset (by ARI):")
    for _, row in best_methods.iterrows():
        logger.info(f"{row['dataset']}: {row['method']} (ARI={row['ari']:.3f})")
    
    logger.info(f"\n✅ Baseline comparison complete! Results saved to: {output_path}")
    logger.info(f"Total experiments run: {len(all_results)}")

if __name__ == "__main__":
    main()
