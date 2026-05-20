#!/usr/bin/env python3
"""
Embedding Method Comparison Script
=================================

Compare DeepCluster vs SupCon embedding quality across datasets.
Evaluates:
- k-NN accuracy on embeddings
- Clustering performance with same method
- Embedding space visualization
- Feature importance analysis

Usage:
    python experiments/compare_embedding_methods.py --dataset UNSW-NB15
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from baselines.deepcluster_gnn import DeepClusterTrainer
from hgnn.hgnn_correlation import HGNNCorrelationEngine
from training.train_on_datasets import PublicDatasetGraphConverter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_supcon_embeddings(dataset_path: str, checkpoint_path: str, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load SupCon-trained embeddings from checkpoint."""
    logger.info(f"Loading SupCon embeddings from {checkpoint_path}")
    
    # Load dataset
    df = pd.read_parquet(dataset_path) if dataset_path.endswith('.parquet') else pd.read_csv(dataset_path)
    
    # Load test indices if available
    test_indices_path = checkpoint_path.replace('best.pt', 'test_indices.npy')
    if os.path.exists(test_indices_path):
        test_indices = np.load(test_indices_path)
        df = df.iloc[test_indices].reset_index(drop=True)
    
    # Convert to graph
    converter = PublicDatasetGraphConverter()
    graph = converter.convert(df)
    
    # Load SupCon model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize HGNN with SupCon weights
    hgnn = HGNNCorrelationEngine(
        model_path=checkpoint_path,
        device=device,
        pure_unsupervised=True
    )
    
    # Extract embeddings
    with torch.no_grad():
        embeddings, _ = hgnn.correlate(graph)
    
    labels = df['campaign_id'].values if 'campaign_id' in df.columns else df['label'].values
    
    return embeddings.cpu().numpy(), labels, df.index.values


def load_deepcluster_embeddings(dataset_path: str, checkpoint_path: str, device: str = 'cuda') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load DeepCluster embeddings."""
    logger.info(f"Loading DeepCluster embeddings using {checkpoint_path}")
    
    # Load dataset
    df = pd.read_parquet(dataset_path) if dataset_path.endswith('.parquet') else pd.read_csv(dataset_path)
    
    # Sample if dataset is too large
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled to {len(df)} records")
    
    # Initialize DeepCluster trainer
    trainer = DeepClusterTrainer(
        checkpoint_path=checkpoint_path,
        n_clusters=len(df['campaign_id'].unique()) if 'campaign_id' in df.columns else len(df['label'].unique()),
        device=device
    )
    
    # Extract embeddings
    embeddings = trainer.extract_embeddings_batched(df)
    labels = df['campaign_id'].values if 'campaign_id' in df.columns else df['label'].values
    
    return embeddings, labels, df.index.values


def evaluate_embedding_quality(embeddings: np.ndarray, labels: np.ndarray, method_name: str) -> Dict:
    """Evaluate embedding quality using multiple metrics."""
    logger.info(f"Evaluating {method_name} embeddings...")
    
    results = {
        'method': method_name,
        'n_samples': len(embeddings),
        'n_features': embeddings.shape[1],
        'n_classes': len(np.unique(labels))
    }
    
    # k-NN accuracy
    try:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn_scores = cross_val_score(knn, embeddings, labels, cv=min(5, len(set(labels))), scoring='accuracy')
        results['knn_accuracy'] = knn_scores.mean()
        results['knn_accuracy_std'] = knn_scores.std()
        logger.info(f"k-NN accuracy: {results['knn_accuracy']:.3f} ± {results['knn_accuracy_std']:.3f}")
    except Exception as e:
        logger.warning(f"k-NN evaluation failed: {e}")
        results['knn_accuracy'] = np.nan
    
    # K-Means clustering
    try:
        n_clusters = len(np.unique(labels))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        results['kmeans_ari'] = adjusted_rand_score(labels, cluster_labels)
        results['kmeans_nmi'] = normalized_mutual_info_score(labels, cluster_labels)
        logger.info(f"K-Means ARI: {results['kmeans_ari']:.3f}, NMI: {results['kmeans_nmi']:.3f}")
    except Exception as e:
        logger.warning(f"K-Means evaluation failed: {e}")
        results['kmeans_ari'] = np.nan
        results['kmeans_nmi'] = np.nan
    
    # Embedding statistics
    results['embedding_mean'] = np.mean(embeddings)
    results['embedding_std'] = np.std(embeddings)
    results['embedding_min'] = np.min(embeddings)
    results['embedding_max'] = np.max(embeddings)
    
    return results


def compare_methods_on_dataset(dataset_name: str, dataset_path: str, 
                               supcon_checkpoint: str, deepcluster_checkpoint: str) -> List[Dict]:
    """Compare SupCon vs DeepCluster on a single dataset."""
    logger.info(f"Comparing methods on {dataset_name}")
    
    results = []
    
    # SupCon evaluation
    if os.path.exists(supcon_checkpoint):
        try:
            supcon_emb, supcon_labels, supcon_indices = load_supcon_embeddings(dataset_path, supcon_checkpoint)
            supcon_results = evaluate_embedding_quality(supcon_emb, supcon_labels, "SupCon")
            results.append(supcon_results)
        except Exception as e:
            logger.error(f"SupCon evaluation failed: {e}")
    else:
        logger.warning(f"SupCon checkpoint not found: {supcon_checkpoint}")
    
    # DeepCluster evaluation
    if os.path.exists(deepcluster_checkpoint):
        try:
            dc_emb, dc_labels, dc_indices = load_deepcluster_embeddings(dataset_path, deepcluster_checkpoint)
            dc_results = evaluate_embedding_quality(dc_emb, dc_labels, "DeepCluster")
            results.append(dc_results)
        except Exception as e:
            logger.error(f"DeepCluster evaluation failed: {e}")
    else:
        logger.warning(f"DeepCluster checkpoint not found: {deepcluster_checkpoint}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare embedding methods")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["UNSW-NB15", "NSL-KDD", "CICIDS2017", "TON_IoT", "SQTK_SIEM_kcluster", "OpTC"],
                       help="Dataset to evaluate")
    parser.add_argument("--output", type=str, default="experiments/results/embedding_comparison.csv",
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Dataset configurations
    dataset_configs = {
        "UNSW-NB15": {
            "path": "datasets/unsw_nb15/mitre_format.csv",
            "supcon": "hgnn_checkpoints/unsw_supcon_v7/best.pt",
            "deepcluster": "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
        },
        "NSL-KDD": {
            "path": "datasets/nsl_kdd/mitre_format.csv", 
            "supcon": None,  # Not trained yet
            "deepcluster": "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
        },
        "CICIDS2017": {
            "path": "datasets/CICIDS2017/mitre_format_clean.parquet",
            "supcon": "hgnn_checkpoints/cicids_supcon_v7/best.pt",
            "deepcluster": "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
        },
        "TON_IoT": {
            "path": "datasets/ton_iot/mitre_format.parquet",
            "supcon": None,  # Not trained yet
            "deepcluster": "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
        },
        "SQTK_SIEM_kcluster": {
            "path": "datasets/sqtk_siem_kcluster/mitre_format.parquet",
            "supcon": None,  # Not trained yet
            "deepcluster": "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
        },
        "OpTC": {
            "path": "datasets/optc/mitre_format.parquet",
            "supcon": None,  # Not trained yet
            "deepcluster": "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
        }
    }
    
    config = dataset_configs[args.dataset]
    
    # Compare methods
    results = compare_methods_on_dataset(
        args.dataset, 
        config["path"], 
        config["supcon"], 
        config["deepcluster"]
    )
    
    # Save results
    if results:
        df_results = pd.DataFrame(results)
        df_results['dataset'] = args.dataset
        
        # Append to existing results or create new file
        if os.path.exists(args.output):
            existing_df = pd.read_csv(args.output)
            df_results = pd.concat([existing_df, df_results], ignore_index=True)
        
        df_results.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
        
        # Print comparison
        print(f"\n=== {args.dataset} Embedding Comparison ===")
        for result in results:
            print(f"{result['method']}:")
            print(f"  k-NN Accuracy: {result.get('knn_accuracy', 'N/A'):.3f}")
            print(f"  K-Means ARI: {result.get('kmeans_ari', 'N/A'):.3f}")
            print(f"  K-Means NMI: {result.get('kmeans_nmi', 'N/A'):.3f}")
            print()
    else:
        logger.error("No results to save")


if __name__ == "__main__":
    main()
