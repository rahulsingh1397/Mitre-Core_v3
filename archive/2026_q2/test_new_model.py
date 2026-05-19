#!/usr/bin/env python3
"""Quick test of the new network_v9_v4 model on UNSW-NB15."""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '.')

from hgnn.hgnn_correlation import HGNNCorrelationEngine
from sklearn.metrics import adjusted_rand_score

print('=' * 70)
print('TESTING NEW MODEL: network_v9_no_emb_gpu (Linear Encoder, Fixed Topo Loss)')
print('=' * 70)

# Load the new checkpoint
checkpoint_path = 'hgnn_checkpoints/network_v9_no_emb_gpu/network_it_best.pt'
print(f'\nLoading checkpoint: {checkpoint_path}')

checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
print(f'Checkpoint keys: {list(checkpoint.keys())}')
vocab_info = checkpoint.get('vocab_sizes', 'Not found - using fallback')
print(f'Vocab sizes: {vocab_info}')
print(f'Best loss: {checkpoint.get("loss", "N/A")}')

print('\n' + '=' * 70)
print('LOADING UNSW-NB15 DATA')
print('=' * 70)

# Load UNSW-NB15 (sample for testing)
df = pd.read_csv('datasets/unsw_nb15/mitre_format.csv').sample(n=50000, random_state=42)
print(f'Total alerts: {len(df)}')
print(f'Campaigns: {df["campaign_id"].nunique()}')
print(f'Campaign distribution:')
print(df["campaign_id"].value_counts().head(10))

print('\n' + '=' * 70)
print('RUNNING INFERENCE WITH NEW MODEL')
print('=' * 70)

# Initialize engine with new model
engine = HGNNCorrelationEngine(
    model_path=checkpoint_path,
    device='cuda'
)

# Run correlation
results = engine.correlate(df)

print(f'Predicted clusters: {results["pred_cluster"].nunique()}')
print(f'Results keys: {list(results.keys())}')
print(f'Results type: {type(results)}')
if hasattr(results, 'columns'):
    print(f'Results columns: {list(results.columns)}')
# Calculate ARI manually if needed
if 'ari' not in results:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(results['campaign_id'], results['pred_cluster'])
    nmi = normalized_mutual_info_score(results['campaign_id'], results['pred_cluster'])
    print(f'ARI (manual): {ari:.4f}')
    print(f'NMI (manual): {nmi:.4f}')
else:
    print(f'ARI: {results["ari"]:.4f}')
    print(f'NMI: {results["nmi"]:.4f}')

print('\n' + '=' * 70)
print('COMPARISON WITH BASELINE')
print('=' * 70)
baseline_ari = 0.665
new_ari = results.get("ari", ari)
print(f'Baseline (multidomain_v2):  ARI = {baseline_ari:.4f}')
print(f'New Model (v9_v4):          ARI = {new_ari:.4f}')
print(f'Difference:                 {new_ari - baseline_ari:+.4f}')

if new_ari > baseline_ari:
    print('\n✅ IMPROVEMENT! Learned embeddings are working!')
else:
    print('\n⚠️  Lower than baseline - may need more training or SupCon fine-tuning')
