#!/usr/bin/env python3
"""
Week 4: Lambda ablation on UNSW dataset
Tests (lambda_sup, lambda_pseudo) combinations to identify optimal loss weighting
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

def run_lambda_ablation():
    """Run lambda ablation on UNSW dataset."""
    
    # Base checkpoint
    base_ckpt = "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
    
    # Lambda combinations to test
    lambda_combinations = [
        (1.0, 0.5),  # Default
        (0.5, 0.5),  # Reduced supervised
        (1.0, 0.0),  # No pseudo
        (0.0, 1.0),  # No supervised
        (0.0, 0.0),  # Temporal only
    ]
    
    # Load UNSW dataset only
    dataset_configs = {
        'UNSW-NB15': 'datasets/unsw_nb15/mitre_format.csv',
    }
    
    datasets = load_datasets(dataset_configs)
    label_metadata = load_label_metadata('datasets/label_metadata.csv')
    
    results = []
    
    for lambda_sup, lambda_pseudo in lambda_combinations:
        logger.info(f"Testing lambda_sup={lambda_sup}, lambda_pseudo={lambda_pseudo}")
        
        # Initialize trainer
        trainer = HybridAdaptiveTrainer(
            datasets=datasets,
            label_metadata=label_metadata,
            device='cuda',
            temporal_window_hours=2.0,  # Default window
            lambda_sup=lambda_sup,
            lambda_pseudo=lambda_pseudo,
        )
        
        # Setup model
        trainer.setup_model(base_ckpt)
        
        # Train for 10 epochs (quick ablation)
        start_time = time.time()
        trainer.train(epochs=10, lr=1e-4)
        train_time = time.time() - start_time
        
        # Save checkpoint
        output_dir = f'hgnn_checkpoints/week4_lambda_ablation/UNSW_sup{lambda_sup}_pseudo{lambda_pseudo}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': trainer.hgnn.state_dict(),
            'hidden_dim': trainer.hidden_dim,
            'lambda_sup': lambda_sup,
            'lambda_pseudo': lambda_pseudo,
            'temporal_window_hours': 2.0,
            'train_time_seconds': train_time,
        }, f'{output_dir}/best.pt')
        
        # Record result
        results.append({
            'lambda_sup': lambda_sup,
            'lambda_pseudo': lambda_pseudo,
            'train_time_seconds': train_time,
            'checkpoint_path': f'{output_dir}/best.pt',
        })
        
        logger.info(f"Saved: {output_dir}/best.pt (train_time: {train_time:.1f}s)")
    
    # Save results summary
    results_df = pd.DataFrame(results)
    results_df.to_csv('experiments/results/week4_lambda_ablation_summary.csv', index=False)
    logger.info(f"Lambda ablation complete. Results saved to experiments/results/week4_lambda_ablation_summary.csv")
    
    return results_df

if __name__ == '__main__':
    run_lambda_ablation()
