"""
experiments/run_multisource_experiment.py
-----------------------------------------
Multi-source experiment: UNSW + NSL-KDD merged using MultiSourceIngestionPipeline.
Tests cross-sensor correlation on merged sensor feeds.

Usage:
    python experiments/run_multisource_experiment.py \
        --checkpoint hgnn_checkpoints/multidomain_v2/best_supervised.pt \
        --output experiments/results/multisource_unsw_nslkdd.csv
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.dataset_profiler import MultiSourceIngestionPipeline
from hgnn.hgnn_correlation import HGNNCorrelationEngine
from utils.seed_control import set_seed

set_seed(42)

def run_multisource_experiment(checkpoint_path: str, output_path: str, gate: float = 0.7, sample_size: int = 10000):
    """Run multi-source experiment with UNSW + NSL-KDD merged."""
    
    # Load datasets
    unsw_df = pd.read_csv("datasets/unsw_nb15/mitre_format.csv")
    nsl_df = pd.read_csv("datasets/nsl_kdd/mitre_format.csv")
    
    print(f"Loaded UNSW: {len(unsw_df)} alerts")
    print(f"Loaded NSL-KDD: {len(nsl_df)} alerts")
    
    # Sample if too large (HDBSCAN O(n²) memory)
    if len(unsw_df) > sample_size:
        unsw_df = unsw_df.sample(n=sample_size, random_state=42)
        print(f"Sampled UNSW to {len(unsw_df)} alerts")
    if len(nsl_df) > sample_size:
        nsl_df = nsl_df.sample(n=sample_size, random_state=42)
        print(f"Sampled NSL-KDD to {len(nsl_df)} alerts")
    
    # Create pipeline and merge
    pipeline = MultiSourceIngestionPipeline(temporal_window_hours=1.0)
    pipeline.add_source(unsw_df, source_name="unsw")
    pipeline.add_source(nsl_df, source_name="nsl_kdd")
    merged_df = pipeline.merge()
    
    print(f"Merged dataset: {len(merged_df)} alerts")
    print(f"Data sources: {merged_df['data_source'].value_counts().to_dict()}")
    
    # Prepare ground truth labels for evaluation
    # Combine campaign_id from both sources (they're prefixed with source name during merge)
    merged_df['campaign_id'] = merged_df['campaign_id'].fillna(-1)
    
    # Create synthetic ground truth: treat each source's campaigns as distinct
    # This simulates detecting which source each alert came from
    merged_df['source_label'] = merged_df['data_source'].astype('category').cat.codes
    
    # Initialize engine with cross-sensor features
    engine = HGNNCorrelationEngine(
        model_path=checkpoint_path,
        device="cpu",
        confidence_gate=gate,
        track_data_source=True,
        build_precedes_edges=True,
        precedes_window_hours=2.0,
    )
    
    # Run correlation with multi-source features
    results = engine.correlate(merged_df)
    
    # Evaluate
    y_true = merged_df['source_label'].values
    y_pred = results['pred_cluster'].values
    
    # Calculate metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    
    # Check if clusters align with sources
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix (Sources x Clusters):")
    print(cm)
    
    # Compute purity - how well do clusters separate sources?
    # For each cluster, find the dominant source
    cluster_purities = []
    for cluster_idx in range(cm.shape[1]):
        cluster_counts = cm[:, cluster_idx]
        if cluster_counts.sum() > 0:
            purity = cluster_counts.max() / cluster_counts.sum()
            cluster_purities.append(purity)
    
    avg_purity = np.mean(cluster_purities) if cluster_purities else 0
    
    print(f"\n--- Multi-Source Experiment Results ---")
    print(f"Total alerts: {len(merged_df)}")
    print(f"Source distribution: {merged_df['data_source'].value_counts().to_dict()}")
    print(f"ARI (source detection): {ari:.4f}")
    print(f"NMI (source detection): {nmi:.4f}")
    print(f"Cluster purity: {avg_purity:.4f}")
    print(f"Number of clusters: {len(set(y_pred))}")
    
    # Save results
    results_df = pd.DataFrame([{
        'experiment': 'multisource_unsw_nslkdd',
        'total_alerts': len(merged_df),
        'unsw_alerts': (merged_df['data_source'] == 'unsw').sum(),
        'nslkdd_alerts': (merged_df['data_source'] == 'nsl_kdd').sum(),
        'ari': ari,
        'nmi': nmi,
        'cluster_purity': avg_purity,
        'n_clusters': len(set(y_pred)),
        'gate': gate,
        'checkpoint': checkpoint_path,
    }])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    return results_df

def main():
    parser = argparse.ArgumentParser(description="Multi-source experiment: UNSW + NSL-KDD")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to HGNN checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument("--gate", type=float, default=0.7, help="Gate threshold")
    parser.add_argument("--sample_size", type=int, default=10000, help="Max samples per source")
    args = parser.parse_args()
    
    run_multisource_experiment(args.checkpoint, args.output, args.gate, args.sample_size)

if __name__ == "__main__":
    main()
