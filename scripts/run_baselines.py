"""
scripts/run_baselines.py
-------------------------
Baseline clustering comparison for HGNN paper.
Runs HDBSCAN and KMeans directly on raw alert features (no GNN).
"""

import pandas as pd
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import logging
import time
from pathlib import Path
import argparse
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hgnn.hgnn_correlation import AlertToGraphConverter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_preprocess_data(dataset_config):
    """Load and preprocess dataset for baseline clustering."""
    logger.info(f"Loading {dataset_config['name']} dataset...")
    
    # Load dataset
    if dataset_config['path'].endswith('.parquet'):
        df = pd.read_parquet(dataset_config['path'])
    else:
        df = pd.read_csv(dataset_config['path'])
    
    # Sample if specified
    if dataset_config.get('sample_size') and len(df) > dataset_config['sample_size']:
        df = df.sample(n=dataset_config['sample_size'], random_state=42).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} records")
    
    # Extract ground truth labels
    true_labels = LabelEncoder().fit_transform(
        df[dataset_config['label_col']].fillna('UNKNOWN').astype(str)
    )
    
    # Create feature matrix from alert data
    converter = AlertToGraphConverter()
    
    # Extract raw features (same as HGNN would see)
    features = []
    for _, row in df.iterrows():
        # Create feature vector from alert attributes
        feature_vector = []
        
        # Basic alert features
        feature_vector.append(hash(str(row.get('ProcessName', ''))) % 1000)
        feature_vector.append(hash(str(row.get('alert_type', ''))) % 1000)
        feature_vector.append(hash(str(row.get('SourceUserName', ''))) % 1000)
        feature_vector.append(hash(str(row.get('SourceHostName', ''))) % 1000)
        feature_vector.append(hash(str(row.get('SourceAddress', ''))) % 1000)
        feature_vector.append(hash(str(row.get('DestinationAddress', ''))) % 1000)
        
        # Temporal features (timestamp)
        if 'Timestamp' in row and pd.notna(row['Timestamp']):
            feature_vector.append(float(row['Timestamp']))
        else:
            feature_vector.append(0.0)
        
        features.append(feature_vector)
    
    X = np.array(features)
    
    # Handle missing values and scale
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of unique clusters: {len(np.unique(true_labels))}")
    
    return X, true_labels, df

def run_hdbscan(X, min_cluster_size=5, min_samples=5):
    """Run HDBSCAN clustering."""
    logger.info(f"Running HDBSCAN with min_cluster_size={min_cluster_size}...")
    
    start_time = time.perf_counter()
    clusterer = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    pred_labels = clusterer.fit_predict(X)
    runtime = time.perf_counter() - start_time
    
    # Handle noise points (label = -1)
    n_clusters = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
    n_noise = list(pred_labels).count(-1)
    
    logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points, {runtime:.3f}s")
    
    return pred_labels, runtime, n_clusters, n_noise

def run_kmeans(X, n_clusters):
    """Run KMeans clustering."""
    logger.info(f"Running KMeans with n_clusters={n_clusters}...")
    
    start_time = time.perf_counter()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X)
    runtime = time.perf_counter() - start_time
    
    logger.info(f"KMeans: {n_clusters} clusters, {runtime:.3f}s")
    
    return pred_labels, runtime, n_clusters

def evaluate_clustering(true_labels, pred_labels, method_name):
    """Evaluate clustering performance."""
    # Filter out noise points for HDBSCAN
    valid_mask = pred_labels != -1
    if not np.any(valid_mask):
        return {'ari': 0.0, 'nmi': 0.0, 'n_valid': 0}
    
    ari = adjusted_rand_score(true_labels[valid_mask], pred_labels[valid_mask])
    nmi = normalized_mutual_info_score(
        true_labels[valid_mask], pred_labels[valid_mask], average_method="arithmetic"
    )
    
    return {
        'ari': ari,
        'nmi': nmi,
        'n_valid': valid_mask.sum(),
        'n_total': len(true_labels),
        'n_noise': (~valid_mask).sum()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='experiments/results/baseline_results.csv', 
                       help='Output CSV path')
    parser.add_argument('--datasets', nargs='*', default=None,
                       help='Specific datasets to test')
    args = parser.parse_args()
    
    # Dataset configurations
    dataset_configs = {
        'UNSW-NB15': {
            'path': 'datasets/UNSW-NB15/mitre_format.parquet',
            'label_col': 'campaign_id',
            'sample_size': 10000,
            'name': 'UNSW-NB15'
        },
        'BETH': {
            'path': 'datasets/BETH/mitre_format.parquet',
            'label_col': 'campaign_id',
            'sample_size': 10000,
            'name': 'BETH'
        },
        'OpTC': {
            'path': 'datasets/DARPA_OpTC/processed_optc_with_campaigns.csv',
            'label_col': 'CampaignId',
            'sample_size': 500,  # Small dataset
            'name': 'OpTC'
        }
    }
    
    # Filter datasets if specified
    if args.datasets:
        dataset_configs = {k: v for k, v in dataset_configs.items() if k in args.datasets}
    
    results = []
    
    for dataset_name, config in dataset_configs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name}")
        
        try:
            # Load and preprocess data
            X, true_labels, df = load_and_preprocess_data(config)
            n_true_clusters = len(np.unique(true_labels))
            
            # HDBSCAN with different parameters
            hdbscan_params = [
                {'min_cluster_size': 5, 'min_samples': 5},
                {'min_cluster_size': 10, 'min_samples': 5},
                {'min_cluster_size': 15, 'min_samples': 5},
            ]
            
            for params in hdbscan_params:
                pred_labels, runtime, n_clusters, n_noise = run_hdbscan(X, **params)
                metrics = evaluate_clustering(true_labels, pred_labels, 'HDBSCAN')
                
                result = {
                    'dataset': dataset_name,
                    'method': 'HDBSCAN',
                    'min_cluster_size': params['min_cluster_size'],
                    'min_samples': params['min_samples'],
                    'n_clusters_pred': n_clusters,
                    'n_clusters_true': n_true_clusters,
                    'n_noise': n_noise,
                    'ari': metrics['ari'],
                    'nmi': metrics['nmi'],
                    'runtime_s': runtime,
                    'n_valid': metrics['n_valid'],
                    'n_total': metrics['n_total']
                }
                results.append(result)
                
                logger.info(f"  ARI: {metrics['ari']:.4f}, NMI: {metrics['nmi']:.4f}")
            
            # KMeans with true number of clusters
            pred_labels, runtime, n_clusters = run_kmeans(X, n_true_clusters)
            metrics = evaluate_clustering(true_labels, pred_labels, 'KMeans')
            
            result = {
                'dataset': dataset_name,
                'method': 'KMeans',
                'min_cluster_size': None,
                'min_samples': None,
                'n_clusters_pred': n_clusters,
                'n_clusters_true': n_true_clusters,
                'n_noise': 0,
                'ari': metrics['ari'],
                'nmi': metrics['nmi'],
                'runtime_s': runtime,
                'n_valid': metrics['n_valid'],
                'n_total': metrics['n_total']
            }
            results.append(result)
            
            logger.info(f"  ARI: {metrics['ari']:.4f}, NMI: {metrics['nmi']:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_name}: {e}")
            continue
    
    # Save results
    results_df = pd.DataFrame(results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(args.output, index=False)
    
    logger.info(f"\nBaseline evaluation complete! Results saved to {args.output}")
    logger.info("\nSummary:")
    summary = results_df.groupby(['dataset', 'method'])['ari'].max().unstack()
    print(summary)

if __name__ == "__main__":
    main()
