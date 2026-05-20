# DEPRECATED: Epsilon sweep for network_v4 (pre-v9_v3 era).
# Superseded by run_gate_tuning.py. Do not use.
# DEPRECATED: superseded by run_gate_tuning.py.
"""
experiments/run_epsilon_sweep.py
--------------------------------
Cluster selection epsilon sweep for network_v4 checkpoint.
Tests epsilon values [0.05, 0.1, 0.15, 0.2, 0.3] to find optimal cluster merging.

Usage:
    python experiments/run_epsilon_sweep.py \
        --checkpoint hgnn_checkpoints/network_v4/unsw_nb15_best.pt \
        --dataset UNSW-NB15 \
        --output experiments/results/epsilon_sweep_network_v4.csv

Expected: Find epsilon that gives 8-15 clusters for UNSW with ARI > 0.35
"""

import argparse
import time
import logging
import sys
import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hgnn.hgnn_correlation import HGNNCorrelationEngine
from utils.seed_control import set_seed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("epsilon_sweep")

# Epsilon values to test
EPSILON_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

# Dataset configurations matching run_gate_tuning.py
DATASET_CONFIG = {
    "UNSW-NB15": {
        "path": "datasets/unsw_nb15/mitre_format.csv",
        "label_col": "campaign_id",
        "hdbscan_min_cluster_size": 30,
        "hdbscan_pca_components": 16,
        "use_geometric_confidence": True,
        "hdbscan_auto_tune": True,
        "use_umap": True,
        "umap_n_components": 10,
        "sample_size": 10000,  # ADDED: Limit to avoid memory issues
        "stratified_sample": True,
        "expected_clusters": (8, 15),  # Range for optimal clustering
    },
    "NSL-KDD": {
        "path": "datasets/nsl_kdd/mitre_format.csv",
        "label_col": "tactic",
        "hdbscan_min_cluster_size": 5,
        "hdbscan_pca_components": 16,
        "use_geometric_confidence": True,
        "hdbscan_auto_tune": True,
        "expected_clusters": (10, 20),
    },
    "TON_IoT": {
        "path": "datasets/TON_IoT/mitre_format.parquet",
        "label_col": "campaign_id",
        "hdbscan_min_cluster_size": 15,
        "hdbscan_pca_components": 16,
        "sample_size": 10000,
        "stratified_sample": True,
        "use_geometric_confidence": True,
        "hdbscan_auto_tune": True,
        "use_umap": True,
        "umap_n_components": 10,
        "expected_clusters": (10, 25),
    },
}


def load_dataset(path: str, sample_size: int = None, stratified: bool = False, 
                 label_col: str = None, dataset_name: str = None) -> pd.DataFrame:
    """Load a preprocessed dataset."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Preprocessed dataset not found: {path}")
    
    if p.suffix == ".csv":
        df = pd.read_csv(p, low_memory=False)
    elif p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")
    
    # Remap column names
    col_map = {
        "src_ip": "SourceAddress",
        "dst_ip": "DestinationAddress",
        "hostname": "SourceHostName",
        "username": "SourceUserName",
        "timestamp": "EndDate",
        "alert_type": "MalwareIntelAttackType",
        "tactic": "AttackTechnique",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    
    # Sample if needed
    if sample_size is not None and len(df) > sample_size:
        if stratified and label_col and label_col in df.columns:
            df = (
                df.groupby(label_col, group_keys=False)
                  .apply(lambda g: g.sample(
                      min(len(g), max(1, int(sample_size * len(g) / len(df)))),
                      random_state=99
                  ))
            )
            df = df.sample(frac=1, random_state=99).reset_index(drop=True)
            logger.info(f"Stratified sampling: {len(df)} records")
        else:
            df = df.sample(n=sample_size, random_state=99).reset_index(drop=True)
            logger.info(f"Random sampling: {len(df)} records")
    
    return df


def encode_labels(df: pd.DataFrame, label_col: str) -> np.ndarray:
    """Encode ground-truth labels to integers."""
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")
    le = LabelEncoder()
    return le.fit_transform(df[label_col].fillna("UNKNOWN").astype(str).values)


def run_epsilon_sweep(checkpoint_path: str, dataset_name: str, output_path: str) -> None:
    """Run epsilon sweep for a single dataset."""
    set_seed(42)
    logger.info(f"Starting epsilon sweep for {dataset_name}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    
    config = DATASET_CONFIG.get(dataset_name)
    if not config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Load dataset
    df = load_dataset(
        config["path"], 
        sample_size=config.get("sample_size"),
        stratified=config.get("stratified_sample", False),
        label_col=config.get("label_col"),
        dataset_name=dataset_name
    )
    true_labels = encode_labels(df, config["label_col"])
    
    logger.info(f"Loaded {len(df)} records, {len(np.unique(true_labels))} true classes")
    
    results = []
    best_ari = 0.0
    best_epsilon = 0.0
    best_config = None
    
    for epsilon in EPSILON_VALUES:
        logger.info(f"\nTesting epsilon={epsilon:.2f}")
        
        engine = HGNNCorrelationEngine(
            model_path=checkpoint_path,
            device="cpu",
            use_geometric_confidence=config.get("use_geometric_confidence", True),
            pure_unsupervised=True,
            hdbscan_min_cluster_size=config["hdbscan_min_cluster_size"],
            hdbscan_pca_components=config["hdbscan_pca_components"],
            hdbscan_auto_tune=config.get("hdbscan_auto_tune", False),
            hdbscan_cluster_selection_epsilon=epsilon,
            hdbscan_use_umap=config.get("use_umap", False),
            hdbscan_umap_n_components=config.get("umap_n_components", 10),
        )
        
        t_start = time.perf_counter()
        result_df = engine.correlate(df)
        latency = time.perf_counter() - t_start
        
        pred_labels = result_df["pred_cluster"].values
        n_clusters = result_df["pred_cluster"].nunique()
        
        # Calculate metrics
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels, average_method="arithmetic")
        
        # Binary ARI for 2-class datasets
        n_true_classes = df[config["label_col"]].nunique()
        if n_true_classes == 2:
            cluster_majority = (
                result_df.groupby("pred_cluster")[config["label_col"]]
                .agg(lambda x: x.mode().iloc[0])
            )
            binary_pred = result_df["pred_cluster"].map(cluster_majority)
            binary_ari = float(adjusted_rand_score(result_df[config["label_col"]], binary_pred))
        else:
            binary_ari = float("nan")
        
        # Check if clusters are in expected range
        expected_min, expected_max = config.get("expected_clusters", (5, 50))
        in_range = expected_min <= n_clusters <= expected_max
        
        row = {
            "dataset": dataset_name,
            "checkpoint": checkpoint_path,
            "epsilon": epsilon,
            "ari": ari,
            "binary_ari": binary_ari,
            "nmi": nmi,
            "n_clusters": n_clusters,
            "n_true_classes": n_true_classes,
            "in_expected_range": in_range,
            "latency_s": latency,
            "avg_confidence": float(result_df["cluster_confidence"].mean()),
        }
        results.append(row)
        
        status = "✓" if in_range else "✗"
        logger.info(
            f"  {status} epsilon={epsilon:.2f}: ARI={ari:.4f}, clusters={n_clusters}, "
            f"expected=({expected_min}-{expected_max})"
        )
        
        # Track best configuration
        if ari > best_ari and in_range:
            best_ari = ari
            best_epsilon = epsilon
            best_config = row.copy()
        
        del result_df, engine
        gc.collect()
    
    # Save results
    results_df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Epsilon sweep complete. Results saved to {output_path}")
    
    if best_config:
        logger.info(f"\nBEST CONFIGURATION:")
        logger.info(f"  epsilon={best_epsilon:.2f}")
        logger.info(f"  ARI={best_ari:.4f}")
        logger.info(f"  clusters={best_config['n_clusters']}")
    else:
        logger.warning("No configuration found within expected cluster range!")
        # Find best overall even if out of range
        best_idx = results_df["ari"].idxmax()
        best = results_df.iloc[best_idx]
        logger.info(f"\nBest overall (out of range): epsilon={best['epsilon']:.2f}, ARI={best['ari']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to network_v4 checkpoint")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=list(DATASET_CONFIG.keys()),
                       help="Dataset to test")
    parser.add_argument("--output", type=str, 
                       default="experiments/results/epsilon_sweep_results.csv",
                       help="Output CSV path")
    args = parser.parse_args()
    
    run_epsilon_sweep(args.checkpoint, args.dataset, args.output)


if __name__ == "__main__":
    main()
