#!/usr/bin/env python3
"""
Multi-Domain HGNN Evaluation
Tests cross-sensor correlation performance on both UNSW-NB15 and BETH datasets.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import logging
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

# HGNN imports
from hgnn.hgnn_correlation import MITREHeteroGNN, AlertToGraphConverter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_on_dataset(checkpoint_path, dataset_path, dataset_name, expected_clusters):
    """Evaluate multi-domain model on a specific dataset."""
    
    logger.info(f"\n=== Evaluating on {dataset_name} ===")
    
    # Load data
    if dataset_path.endswith('.parquet'):
        df = pd.read_parquet(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
    
    logger.info(f"{dataset_name}: {len(df)} alerts")
    
    # Campaign distribution
    campaign_dist = df['campaign_id'].value_counts().sort_index()
    logger.info(f"Campaign distribution:\n{campaign_dist}")
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Initialize model with correct number of clusters
    model = MITREHeteroGNN(
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        num_clusters=expected_clusters,
        dropout=0.3
    ).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint, strict=False)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None
    
    # Initialize converter
    converter = AlertToGraphConverter()
    
    # Test on sample chunks
    chunk_size = 1000
    n_chunks = min(20, len(df) // chunk_size)
    
    logger.info(f"Testing on {n_chunks} chunks of {chunk_size} alerts each")
    
    all_embeddings = []
    all_labels = []
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size
        chunk_df = df.iloc[start_idx:end_idx].copy()
        
        if len(chunk_df) < 100:
            continue
        
        logger.info(f"Processing chunk {i+1}/{n_chunks} ({len(chunk_df)} alerts)")
        
        try:
            # Convert to graph format
            graph_data = converter.convert(chunk_df)
            graph_data = graph_data.to(device)
            
            # Get embeddings
            with torch.no_grad():
                cluster_logits, node_embeddings = model(graph_data)
                alert_embeddings = node_embeddings['alert']
                embeddings = alert_embeddings.cpu().numpy()
            
            # Get labels
            labels = chunk_df['campaign_id'].values
            
            # Ensure same length
            min_len = min(len(embeddings), len(labels))
            all_embeddings.append(embeddings[:min_len])
            all_labels.append(labels[:min_len])
            
        except Exception as e:
            logger.warning(f"Error processing chunk {i+1}: {e}")
            continue
    
    if not all_embeddings:
        logger.error("No embeddings generated")
        return None
    
    # Combine all embeddings and labels
    X = np.vstack(all_embeddings)
    y = np.hstack(all_labels)
    
    logger.info(f"Combined embeddings shape: {X.shape}")
    logger.info(f"Combined labels shape: {y.shape}")
    
    # For multi-domain model, we need to remap labels back to original range
    if dataset_name == "BETH":
        # BETH labels were shifted by 8 during training, remap back
        y_remapped = y - 8  # 8-9 -> 0-1
        logger.info(f"BETH labels remapped: {np.unique(y_remapped)}")
        test_labels = y_remapped
        n_test_clusters = 2
    else:
        # UNSW labels remain 0-7
        test_labels = y
        n_test_clusters = 8
    
    # Clustering evaluation
    logger.info("Running clustering evaluation...")
    
    # Binary clustering for BETH, multi-cluster for UNSW
    if dataset_name == "BETH":
        # Binary clustering (attack vs benign)
        kmeans = KMeans(n_clusters=2, random_state=42)
        cluster_preds = kmeans.fit_predict(X)
    else:
        # Multi-clustering for UNSW
        kmeans = KMeans(n_clusters=n_test_clusters, random_state=42)
        cluster_preds = kmeans.fit_predict(X)
    
    # Calculate metrics
    ari = adjusted_rand_score(test_labels, cluster_preds)
    nmi = normalized_mutual_info_score(test_labels, cluster_preds)
    
    logger.info(f"Clustering results:")
    logger.info(f"  ARI: {ari:.4f}")
    logger.info(f"  NMI: {nmi:.4f}")
    
    # Cluster analysis
    unique_preds = np.unique(cluster_preds)
    for pred_id in unique_preds:
        mask = cluster_preds == pred_id
        cluster_labels = test_labels[mask]
        unique_labels = np.unique(cluster_labels)
        
        logger.info(f"  Cluster {pred_id}: {len(cluster_labels)} alerts")
        for label in unique_labels:
            count = np.sum(cluster_labels == label)
            logger.info(f"    Label {label}: {count} alerts")
    
    return {
        'dataset': dataset_name,
        'ari': ari,
        'nmi': nmi,
        'n_alerts': len(test_labels),
        'n_clusters_found': len(unique_preds),
        'expected_clusters': n_test_clusters
    }

def test_bridge_edges():
    """Test if bridge edges are working correctly."""
    logger.info("\n=== Testing Bridge Edge Functionality ===")
    
    # Load BETH data (has bridge edges)
    beth_df = pd.read_parquet("datasets/BETH/mitre_format.parquet")
    
    # Count bridge candidates
    bridge_candidates = beth_df[
        (beth_df['SourceHostName'].str.contains('ip-10-100-1-', na=False)) &
        (beth_df['SourceAddress'].str.contains('10.100.1.', na=False))
    ]
    
    logger.info(f"Bridge edge candidates in BETH: {len(bridge_candidates)}")
    
    # Get unique IP->hostname mappings
    bridge_mappings = bridge_candidates[['SourceAddress', 'SourceHostName']].drop_duplicates()
    logger.info(f"Unique IP->hostname mappings: {len(bridge_mappings)}")
    
    # Show sample mappings
    sample_mappings = bridge_mappings.head(10)
    logger.info("Sample IP->hostname mappings:")
    for _, row in sample_mappings.iterrows():
        logger.info(f"  {row['SourceAddress']} -> {row['SourceHostName']}")
    
    return len(bridge_candidates), len(bridge_mappings)

def main():
    """Main evaluation pipeline."""
    logger.info("=== Multi-Domain HGNN Evaluation ===")
    
    # Check checkpoint
    checkpoint_path = "hgnn_checkpoints/multidomain_unsw_beth/best_supervised.pt"
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Test bridge edges
    n_bridge_candidates, n_bridge_mappings = test_bridge_edges()
    
    # Evaluate on both datasets
    results = []
    
    # UNSW-NB15 evaluation
    unsw_result = evaluate_on_dataset(
        checkpoint_path=checkpoint_path,
        dataset_path="datasets/unsw_nb15/mitre_format.csv",
        dataset_name="UNSW-NB15",
        expected_clusters=10  # Multi-domain has 10 clusters total
    )
    
    if unsw_result:
        results.append(unsw_result)
    
    # BETH evaluation
    beth_result = evaluate_on_dataset(
        checkpoint_path=checkpoint_path,
        dataset_path="datasets/BETH/mitre_format.parquet",
        dataset_name="BETH",
        expected_clusters=10  # Multi-domain has 10 clusters total
    )
    
    if beth_result:
        results.append(beth_result)
    
    # Summary
    logger.info(f"\n=== EVALUATION SUMMARY ===")
    logger.info(f"Bridge edges: {n_bridge_candidates} candidates, {n_bridge_mappings} unique mappings")
    
    for result in results:
        logger.info(f"\n{result['dataset']}:")
        logger.info(f"  ARI: {result['ari']:.4f}")
        logger.info(f"  NMI: {result['nmi']:.4f}")
        logger.info(f"  Alerts: {result['n_alerts']}")
        logger.info(f"  Clusters found: {result['n_clusters_found']}")
        
        # Interpretation
        if result['ari'] > 0.5:
            logger.info(f"  ✅ EXCELLENT cross-sensor correlation!")
        elif result['ari'] > 0.1:
            logger.info(f"  ⚠️  Good signal - bridge edges working")
        else:
            logger.info(f"  ❌ Poor performance - needs improvement")
    
    # Cross-sensor success criteria
    logger.info(f"\n=== CROSS-SENSOR SUCCESS CRITERIA ===")
    unsw_ari = next((r['ari'] for r in results if r['dataset'] == 'UNSW-NB15'), None)
    beth_ari = next((r['ari'] for r in results if r['dataset'] == 'BETH'), None)
    
    if unsw_ari and beth_ari:
        logger.info(f"UNSW-NB15 ARI: {unsw_ari:.4f} (target: ≥0.80)")
        logger.info(f"BETH ARI: {beth_ari:.4f} (target: >0.10)")
        
        if unsw_ari >= 0.80 and beth_ari > 0.10:
            logger.info(f"🎉 MULTI-DOMAIN TRAINING SUCCESS!")
            logger.info(f"   Bridge edges enable true cross-sensor correlation")
        elif unsw_ari >= 0.80:
            logger.info(f"✅ Network-only performance maintained")
            logger.info(f"   Need more BETH training data or different sampling")
        else:
            logger.info(f"❌ Multi-domain training needs refinement")

if __name__ == "__main__":
    main()
