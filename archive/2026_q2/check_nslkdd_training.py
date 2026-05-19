import torch

checkpoint = torch.load('hgnn_checkpoints/network_v9_nslkdd/nsl_kdd_best.pt', map_location='cpu')
print('Checkpoint keys:')
for k, v in checkpoint.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: {v.shape}')
    else:
        print(f'  {k}: {v}')

print(f'\nTraining epochs completed: {checkpoint.get("epoch", "unknown")}')
print(f'Best loss: {checkpoint.get("loss", "unknown")}')
