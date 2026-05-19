#!/usr/bin/env python3
"""Check what's in the baseline model."""

import torch
import sys
sys.path.insert(0, '.')

# Load baseline checkpoint
ckpt = torch.load('hgnn_checkpoints/multidomain_v2/best_supervised.pt', map_location='cpu', weights_only=False)
print('Baseline checkpoint keys:', list(ckpt.keys()))

# Check if it has supervised loss components
if 'model_state_dict' in ckpt:
    state_dict = ckpt['model_state_dict']
    print(f'\nModel parameters with "cluster" or "class" in name:')
    for key in state_dict.keys():
        if 'cluster' in key.lower() or 'class' in key.lower() or 'supervis' in key.lower():
            print(f'  {key}: {state_dict[key].shape}')
    
    print(f'\nTotal parameters: {len(state_dict)}')
    print(f'First few keys: {list(state_dict.keys())[:10]}')
