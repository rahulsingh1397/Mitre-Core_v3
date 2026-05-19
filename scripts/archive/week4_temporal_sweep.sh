#!/bin/bash
# Week 4: Temporal window sweep on TON_IoT and SQTK_SIEM
# Tests temporal_window_hours in {0.5, 2, 6, 24} for regression triage

set -e

echo "=== Week 4 Temporal Window Sweep ==="
echo "Target datasets: TON_IoT, SQTK_SIEM (regression triage)"
echo "Window values: 0.5h, 2h, 6h, 24h"

# Base checkpoint
BASE_CKPT="hgnn_checkpoints/network_v9_v3/network_it_best.pt"

# Window values to test
WINDOWS=(0.5 2 6 24)

# Datasets to test
DATASETS=("TON_IoT" "SQTK_SIEM")

for window in "${WINDOWS[@]}"; do
    echo ""
    echo "=== Testing temporal_window_hours=${window} ==="
    
    for dataset in "${DATASETS[@]}"; do
        echo "Training on ${dataset} with window=${window}h..."
        
        # Create dataset-specific config
        python -c "
import sys
sys.path.append('.')
from training.train_hybrid_v10 import load_datasets, load_label_metadata, HybridAdaptiveTrainer
import torch
import logging

# Load single dataset
dataset_configs = {
    '${dataset}': {
        'TON_IoT': 'datasets/TON_IoT/mitre_format.parquet',
        'SQTK_SIEM': 'datasets/SQTK_SIEM/mitre_core_format.csv'
    }['${dataset}']
}

datasets = load_datasets(dataset_configs)
label_metadata = load_label_metadata('datasets/label_metadata.csv')

# Initialize trainer
trainer = HybridAdaptiveTrainer(
    datasets=datasets,
    label_metadata=label_metadata,
    device='cuda',
    temporal_window_hours=${window},
    lambda_sup=1.0,
    lambda_pseudo=0.5,
)

# Setup model
trainer.setup_model('${BASE_CKPT}')

# Train for 10 epochs (quick sweep)
trainer.train(epochs=10, lr=1e-4)

# Save checkpoint
import os
output_dir = f'hgnn_checkpoints/week4_temporal_sweep/${dataset}_window_${window}h'
os.makedirs(output_dir, exist_ok=True)
torch.save({
    'model_state_dict': trainer.hgnn.state_dict(),
    'hidden_dim': trainer.hidden_dim,
    'temporal_window_hours': ${window},
    'lambda_sup': 1.0,
    'lambda_pseudo': 0.5,
}, f'{output_dir}/best.pt')

print(f'Saved: {output_dir}/best.pt')
"
        
        echo "Completed ${dataset} with window=${window}h"
    done
done

echo ""
echo "=== Temporal sweep complete ==="
echo "Next: Run gate evaluation on all checkpoints"
