#!/usr/bin/env python3
"""
Debug script to trace BETH domain head inference issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score

from hgnn.hgnn_correlation import HGNNCorrelationEngine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_beth_inference():
    """Debug BETH domain head loading and inference"""
    
    checkpoint_path = "hgnn_checkpoints/multidomain_v2_beth_finetuned/best_supervised.pt"
    dataset_path = "datasets/BETH/mitre_format.parquet"
    
    logger.info("=" * 70)
    logger.info("DEBUGGING BETH DOMAIN HEAD")
    logger.info("=" * 70)
    
    # 1. Check checkpoint contents
    logger.info("\n[1] CHECKPOINT CONTENTS")
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    logger.info(f"  Domain saved in ckpt: {ckpt.get('domain', 'N/A')}")
    logger.info(f"  Val ARI from training: {ckpt.get('val_ari', 'N/A')}")
    logger.info(f"  Class weights: {ckpt.get('class_weights', 'N/A')}")
    
    state_dict = ckpt['model_state_dict']
    domain_head_keys = [k for k in state_dict.keys() if 'domain_heads' in k]
    logger.info(f"  Domain head keys: {domain_head_keys}")
    
    # 2. Load dataset
    logger.info("\n[2] DATASET LOADING")
    df = pd.read_parquet(dataset_path)
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    # Sample for quick testing
    df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    logger.info(f"  Sampled to: {len(df)} records")
    
    # Check label distribution
    if 'campaign_id' in df.columns:
        label_counts = df['campaign_id'].value_counts()
        logger.info(f"  Campaign distribution: {label_counts.to_dict()}")
        logger.info(f"  Class imbalance ratio: {label_counts.max() / label_counts.min():.1f}:1")
    
    # 3. Initialize engine
    logger.info("\n[3] ENGINE INITIALIZATION")
    engine = HGNNCorrelationEngine(
        model_path=checkpoint_path,
        confidence_gate=0.5,
        device='cpu',
        use_geometric_confidence=False,
        hdbscan_min_cluster_size=20,
        hdbscan_pca_components=16,
    )
    
    # Check if domain heads exist
    if engine.model.domain_heads:
        logger.info(f"  Domain heads loaded: {list(engine.model.domain_heads.keys())}")
        for domain, head in engine.model.domain_heads.items():
            # Get output dimension (last layer)
            for name, param in head.named_parameters():
                if '3.weight' in name:  # Last linear layer
                    logger.info(f"  Domain '{domain}' output classes: {param.shape[0]}")
    else:
        logger.error("  NO DOMAIN HEADS LOADED!")
    
    # 4. Test inference with explicit domain
    logger.info("\n[4] INFERENCE TEST")
    
    # Set dataset name for domain detection
    df._dataset_name = "BETH_finetuned"
    
    # Try inference with chunking
    chunk_size = 1000
    results = []
    for i, start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[start:start+chunk_size].copy().reset_index(drop=True)
        chunk._dataset_name = "BETH_finetuned"
        result = engine.correlate(chunk)
        results.append(result)
        logger.info(f"  Chunk {i+1}: {len(result)} predictions")
        
        # Check predictions
        n_clusters = result['pred_cluster'].nunique()
        avg_conf = result['cluster_confidence'].mean()
        logger.info(f"    Clusters: {n_clusters}, Avg confidence: {avg_conf:.4f}")
        
        # Show prediction distribution
        pred_counts = result['pred_cluster'].value_counts()
        logger.info(f"    Prediction distribution: {pred_counts.to_dict()}")
        
        # Check ground truth alignment
        if 'campaign_id' in result.columns:
            true_labels = LabelEncoder().fit_transform(result['campaign_id'].astype(str))
            pred_labels = result['pred_cluster'].values
            try:
                ari = adjusted_rand_score(true_labels, pred_labels)
                logger.info(f"    ARI for this chunk: {ari:.4f}")
            except Exception as e:
                logger.error(f"    ARI calculation failed: {e}")
        
        # Only test first chunk for debugging
        break
    
    logger.info("\n[5] DOMAIN ROUTING ANALYSIS")
    # Manually test what domain would be selected
    dataset_name = "beth_finetuned"
    if engine.model.domain_heads:
        available_heads = list(engine.model.domain_heads.keys())
        logger.info(f"  Available domain heads: {available_heads}")
        logger.info(f"  Dataset name: {dataset_name}")
        
        # Check exact match
        if dataset_name in available_heads:
            logger.info(f"  ✓ Exact match found: {dataset_name}")
        else:
            # Check substring match
            matched = None
            ds_lower = dataset_name.lower()
            for head_key in available_heads:
                if head_key in ds_lower or ds_lower in head_key:
                    matched = head_key
                    break
            if matched:
                logger.info(f"  ✓ Substring match: '{dataset_name}' → '{matched}'")
            else:
                logger.warning(f"  ✗ No domain head match found! Using shared classifier")
    
    logger.info("\n" + "=" * 70)
    logger.info("DEBUG COMPLETE")
    logger.info("=" * 70)

if __name__ == "__main__":
    debug_beth_inference()
