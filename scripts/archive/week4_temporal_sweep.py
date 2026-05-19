#!/usr/bin/env python3
"""
Week 4: Temporal window sweep on TON_IoT and SQTK_SIEM
Tests temporal_window_hours in {0.5, 2, 6, 24} for regression triage
"""

import os
import sys
import time
import logging
from pathlib import Path
import torch
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_hybrid_v10 import load_datasets, load_label_metadata, HybridAdaptiveTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_temporal_sweep(smoke_test=False):
    """Run temporal window sweep on TON_IoT and SQTK_SIEM."""
    
    # Base checkpoint
    base_ckpt = "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
    
    # Test on TON_IoT and SQTK_SIEM
    test_datasets = ['TON_IoT', 'SQTK_SIEM']
    temporal_windows = [0.5] if smoke_test else [0.5, 2.0, 6.0, 24.0]  # hours
    epochs = 1 if smoke_test else 10
    
    # Load label metadata
    label_metadata = load_label_metadata('datasets/label_metadata.csv')
    
    results = []
    
    for dataset_name in test_datasets:
        logger.info(f"=== Testing dataset: {dataset_name} ===")
        
        # Load dataset
        dataset_configs = {dataset_name: f'datasets/{dataset_name.lower()}/mitre_format.parquet'}
        datasets = load_datasets(dataset_configs)
        
        if dataset_name not in datasets:
            logger.warning(f"Dataset {dataset_name} not found, skipping")
            continue
        
        for window_hours in temporal_windows:
            logger.info(f"Testing {dataset_name} with window={window_hours}h")
            
            # Create output directory
            output_dir = Path(f'hgnn_checkpoints/week4_temporal_sweep/{dataset_name}_window_{window_hours}h')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize trainer
            trainer = HybridAdaptiveTrainer(
                datasets=datasets,
                label_metadata=label_metadata,
                device='cuda',
                temporal_window_hours=window_hours,
                pseudo_confidence_threshold=0.8,
                lambda_sup=1.0,
                lambda_pseudo=0.5,
            )
            
            # Setup model
            trainer.setup_model(base_ckpt)
            
            # Train
            start_time = time.time()
            trainer.train(epochs=epochs, lr=1e-4)
            train_time = time.time() - start_time
            
            # Save checkpoint
            checkpoint_path = output_dir / 'best.pt'
            torch.save({
                'model_state_dict': trainer.hgnn.state_dict(),
                'hidden_dim': trainer.hidden_dim,
                'temperature': trainer.temperature,
                'sup_temperature': trainer.sup_temperature,
                'lambda_sup': 1.0,
                'lambda_pseudo': 0.5,
                'temporal_window_hours': window_hours,
                'pseudo_confidence_threshold': 0.8,
            }, checkpoint_path)
            
            logger.info(f"Saved: {checkpoint_path}")
            logger.info(f"  (train_time: {train_time:.1f}s)")
            
            # Record result
            results.append({
                'dataset': dataset_name,
                'temporal_window_hours': window_hours,
                'train_time_seconds': train_time,
                'checkpoint_path': str(checkpoint_path),
            })
            
            if smoke_test:
                logger.info("=== SMOKE TEST COMPLETE ===")
                break
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiments/results/week4_temporal_sweep_summary.csv', index=False)
    logger.info(f"Temporal sweep complete. Results saved to experiments/results/week4_temporal_sweep_summary.csv")
    
    return results_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Week 4 Temporal Window Sweep")
    parser.add_argument('--smoke', action='store_true', help='Run smoke test (1 epoch, 1 window)')
    args = parser.parse_args()
    
    run_temporal_sweep(smoke_test=args.smoke)
