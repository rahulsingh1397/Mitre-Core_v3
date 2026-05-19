"""
Flow-feature-only baseline for UNSW-NB15 and CICIDS2017.
Uses packet/flow stats directly — no GNN, no graph construction.
Compares K-Means, GMM, HDBSCAN against HGNN ARI.
"""
import pandas as pd
import numpy as np
import argparse
import time
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
import json
import hdbscan
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

from hgnn.hgnn_correlation import HGNNCorrelationEngine, AlertToGraphConverter
from utils.clustering import evaluate_clustering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_features(dataset_path: str, label_col: str, n_true_classes: int):
    # Load data
    logger.info(f"Loading {dataset_path}")
    if dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
        
    labels = df[label_col].fillna("UNKNOWN").astype(str).values
    
    results = []
    
    # 1. Baseline: mean aggregation
    logger.info("Evaluating baseline (mean aggregation, standard features)")
    engine_mean = HGNNCorrelationEngine(
        hidden_dim=128, 
        num_heads=4, 
        num_layers=1, 
        aggr_method='mean',
        device='cpu', # Use CPU for evaluation script to avoid OOM
        hdbscan_metric_fallback=True, # Prevent OOM from large distance matrices
        use_geometric_confidence=False # Skip slow confidence scoring
    )
    
    start_time = time.time()
    result_mean = engine_mean.correlate(df)
    runtime_mean = time.time() - start_time
    pred_mean = result_mean["pred_cluster"].values
    
    metrics_mean = evaluate_clustering(labels, pred_mean, None, df, label_col, n_true_classes)
    logger.info(f"Baseline (mean): ARI={metrics_mean['ari']:.4f}")
    
    results.append({
        "Method": "Baseline (mean)",
        "ARI": metrics_mean['ari'],
        "NMI": metrics_mean['nmi'],
        "Runtime": runtime_mean
    })
    
    # 2. Track C: max aggregation
    logger.info("Evaluating Track C: max aggregation")
    engine_max = HGNNCorrelationEngine(
        hidden_dim=128, 
        num_heads=4, 
        num_layers=1, 
        aggr_method='max',
        device='cpu',
        hdbscan_metric_fallback=True,
        use_geometric_confidence=False
    )
    
    start_time = time.time()
    result_max = engine_max.correlate(df)
    runtime_max = time.time() - start_time
    pred_max = result_max["pred_cluster"].values
    
    metrics_max = evaluate_clustering(labels, pred_max, None, df, label_col, n_true_classes)
    logger.info(f"Track C (max): ARI={metrics_max['ari']:.4f}")
    
    results.append({
        "Method": "Track C (max)",
        "ARI": metrics_max['ari'],
        "NMI": metrics_max['nmi'],
        "Runtime": runtime_max
    })
    
    # 3. Track B/C: max aggregation + Track B features
    # Since Track B features are already in AlertToGraphConverter, they are active in both above,
    # but we list it explicitly here
    
    df_results = pd.DataFrame(results)
    print("\nResults Summary:")
    print(df_results.to_markdown(index=False))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--label_col", type=str, default="campaign_id")
    parser.add_argument("--n_classes", type=int, default=10)
    args = parser.parse_args()
    
    evaluate_features(args.dataset, args.label_col, args.n_classes)

DATASETS = {
    "UNSW-NB15": {
        "path": "datasets/UNSW_NB15/mitre_format.parquet",
        "label_col": "campaign_id",
        "flow_features": ["src_bytes", "dst_bytes"],   # add more if present in CSV
        "n_true_clusters": 8,
        "hgnn_ari": 0.057,
        "hgnn_ami": None,   # fill after Track A runs
        "sample_size": 10000,
    },
    "CICIDS2017": {
        "path": "datasets/CICIDS2017/mitre_format.parquet",
        "label_col": "campaign_id",
        "flow_features": ["src_bytes", "dst_bytes"],
        "n_true_clusters": 16,
        "hgnn_ari": 0.284,
        "hgnn_ami": None,
        "sample_size": 10000,
    },
    "TON_IoT": {
        "path": "datasets/TON_IoT/mitre_format.parquet",
        "label_col": "campaign_id",
        "flow_features": ["src_bytes", "dst_bytes"],
        "n_true_clusters": 9,
        "hgnn_ari": 0.737,
        "hgnn_ami": None,
        "sample_size": 10000,
    },
    "NSL-KDD": {
        "path": "datasets/NSL_KDD/mitre_format.csv",
        "label_col": "campaign_id",
        "flow_features": ["src_bytes", "dst_bytes"],
        "n_true_clusters": 4,
        "hgnn_ari": 0.722,
        "hgnn_ami": None,
        "sample_size": 10000,
    },
}

results = []

for dataset_name, cfg in DATASETS.items():
    print(f"\n=== {dataset_name} ===")
    # Load data
    if cfg["path"].endswith(".parquet"):
        df = pd.read_parquet(cfg["path"])
    else:
        df = pd.read_csv(cfg["path"])

    # Stratified sample
    df = df.dropna(subset=[cfg["label_col"]])
    if len(df) > cfg["sample_size"]:
        df = df.groupby(cfg["label_col"], group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, cfg["sample_size"] // df[cfg["label_col"]].nunique())),
                               random_state=42)
        ).head(cfg["sample_size"])

    true_labels = df[cfg["label_col"]].astype(str).values

    # Select available flow features
    available_features = [f for f in cfg["flow_features"] if f in df.columns]
    if not available_features:
        # Fall back to numeric columns excluding label
        available_features = df.select_dtypes(include=[np.number]).columns.tolist()
        available_features = [c for c in available_features if c != cfg["label_col"]][:10]

    print(f"Features used: {available_features}")
    X = df[available_features].fillna(0).values
    X = StandardScaler().fit_transform(X)

    n_clusters = cfg["n_true_clusters"]

    for method_name, model in [
        ("KMeans",   KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        ("GMM",      GaussianMixture(n_components=n_clusters, random_state=42)),
        ("BGMM",     BayesianGaussianMixture(n_components=n_clusters * 2, random_state=42)),
        ("HDBSCAN",  hdbscan.HDBSCAN(min_cluster_size=50)),
    ]:
        try:
            labels = model.fit_predict(X)
            ari = adjusted_rand_score(true_labels, labels)
            ami = adjusted_mutual_info_score(true_labels, labels)
            n_found = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"  {method_name:8s}: ARI={ari:.4f}  AMI={ami:.4f}  n_clusters={n_found}")
            results.append({
                "dataset": dataset_name, "method": method_name,
                "ari": ari, "ami": ami, "n_clusters_found": n_found,
                "n_true_clusters": n_clusters,
                "hgnn_ari_reference": cfg["hgnn_ari"],
                "features_used": available_features,
            })
        except Exception as e:
            print(f"  {method_name}: FAILED — {e}")

# Save
out_path = "experiments/results/flow_feature_baseline.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")
