#!/usr/bin/env python3
"""Debug which encoder is being used."""

import torch
import sys
sys.path.insert(0, '.')

from hgnn.hgnn_correlation import HGNNCorrelationEngine

print('=' * 60)
print('DEBUGGING ENCODER TYPE')
print('=' * 60)

# Initialize engine with checkpoint
engine = HGNNCorrelationEngine(
    model_path='hgnn_checkpoints/network_v9_v5_gpu/network_it_best.pt',
    device='cuda'
)

# Check model parameters
print(f'Model alert_feature_dim: {engine.model.alert_feature_dim}')
print(f'Model has alert_encoder: {hasattr(engine.model, "alert_encoder")}')

if hasattr(engine.model, 'alert_encoder'):
    print(f'Alert encoder type: {type(engine.model.alert_encoder)}')
    if hasattr(engine.model.alert_encoder, 'get_vocab_info'):
        print(f'Vocab info: {engine.model.alert_encoder.get_vocab_info()}')
    else:
        print('Alert encoder does not have get_vocab_info method')
else:
    print('Model does not have alert_encoder attribute')
    
# Check if model was reinitialized
print(f'Model initialization parameters:')
print(f'  alert_feature_dim: {engine.model.alert_feature_dim}')
print(f'  hidden_dim: {engine.model.hidden_dim}')

# Try to access the forward method's encoder map
try:
    # Create a small test graph
    import torch_geometric as pyg
    from torch_geometric.data import HeteroData
    
    data = HeteroData()
    data['alert'].x = torch.randn(10, 6)  # 6 features
    data['alert', 'connects', 'alert'].edge_index = torch.tensor([[0, 1], [1, 0]], device='cuda')
    
    # Try forward pass
    with torch.no_grad():
        result = engine.model(data.to('cuda'))
    print('Forward pass successful')
except Exception as e:
    print(f'Forward pass failed: {e}')
