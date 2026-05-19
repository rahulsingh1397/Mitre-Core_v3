#!/usr/bin/env python3
"""Debug the learned embeddings to see if they're meaningful."""

import torch
import sys
sys.path.insert(0, '.')

from hgnn.hgnn_correlation import HGNNCorrelationEngine
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print('=' * 60)
print('DEBUGGING EMBEDDINGS QUALITY')
print('=' * 60)

# Load model
checkpoint_path = 'hgnn_checkpoints/network_v9_v7_gpu/network_it_best.pt'
engine = HGNNCorrelationEngine(
    model_path=checkpoint_path,
    device='cuda'
)

# Load a small sample of data
df = pd.read_csv('datasets/unsw_nb15/mitre_format.csv').sample(n=5000, random_state=42)
print(f'Sample size: {len(df)} alerts')
print(f'Campaigns: {df["campaign_id"].nunique()}')

# Get embeddings directly from the model
print('\nExtracting embeddings...')
with torch.no_grad():
    # Convert to graph
    graph = engine.converter.convert(df)
    graph = graph.to('cuda')
    
    # Get embeddings
    _, embeddings = engine.model(graph)
    alert_embeddings = embeddings['alert'].cpu().numpy()
    print(f'Embeddings shape: {alert_embeddings.shape}')

# Check embedding statistics
print(f'\nEmbedding stats:')
print(f'  Mean: {alert_embeddings.mean():.4f}')
print(f'  Std: {alert_embeddings.std():.4f}')
print(f'  Min: {alert_embeddings.min():.4f}')
print(f'  Max: {alert_embeddings.max():.4f}')

# Check if embeddings vary by campaign
print(f'\nEmbedding variation by campaign:')
for campaign_id in df['campaign_id'].unique():
    mask = df['campaign_id'] == campaign_id
    if mask.sum() > 10:  # Only check campaigns with enough samples
        camp_emb = alert_embeddings[mask]
        print(f'  Campaign {campaign_id}: mean={camp_emb.mean():.4f}, std={camp_emb.std():.4f}, n={mask.sum()}')

# PCA visualization
print(f'\nRunning PCA for visualization...')
pca = PCA(n_components=2)
emb_2d = pca.fit_transform(alert_embeddings)
print(f'PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}')

# Simple clustering check
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, random_state=42)
pred_clusters = kmeans.fit_predict(alert_embeddings)

from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(df['campaign_id'], pred_clusters)
print(f'\nSimple K-means ARI: {ari:.4f}')

# Check if embeddings are collapsed (all similar)
distances = np.linalg.norm(alert_embeddings[:, None] - alert_embeddings[None, :], axis=2)
np.fill_diagonal(distances, np.nan)
mean_distance = np.nanmean(distances)
print(f'Mean pairwise distance: {mean_distance:.4f}')

if mean_distance < 0.1:
    print('WARNING: Embeddings may be collapsed (too similar)')

print('\n' + '=' * 60)
print('EMBEDDING QUALITY SUMMARY')
print('=' * 60)
print(f'1. Embeddings vary: {"YES" if alert_embeddings.std() > 0.01 else "NO"}')
print(f'2. Campaign separation: {"SOME" if ari > 0 else "NONE"}')
print(f'3. Embedding collapse: {"LIKELY" if mean_distance < 0.1 else "UNLIKELY"}')
print(f'4. PCA variance captured: {pca.explained_variance_ratio_.sum():.1%}')
