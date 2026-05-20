#!/usr/bin/env python3
# EXPERIMENTAL: This file contains early research explorations (VGAE link-prediction)
# and is not part of the production pipeline.
"""
Train MITREHeteroGNN using Variational Graph Autoencoder (VGAE) Link Prediction
"""

import logging
import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch_geometric.data import HeteroData
from tqdm import tqdm

from utils.seed_control import set_seed
from hgnn.hgnn_correlation import MITREHeteroGNN
from training.train_on_datasets import PublicDatasetGraphConverter
from hgnn.vgae_pretraining import HeteroVGAE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_vgae")

def get_positive_edges(graph: HeteroData) -> torch.Tensor:
    """Extract all intra-alert positive edges (shares_ip, shares_host, temporal_near)"""
    edge_list = []
    for et in graph.edge_types:
        src_type, rel, dst_type = et
        if src_type == 'alert' and dst_type == 'alert' and et in graph.edge_index_dict:
            edge_list.append(graph.edge_index_dict[et])
    if edge_list:
        return torch.cat(edge_list, dim=1)
    return torch.empty((2, 0), dtype=torch.long, device=graph['alert'].x.device)

def sample_negative_edges(pos_edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Sample random negative edges (i, j) that are not in pos_edge_index"""
    num_neg = pos_edge_index.size(1)
    if num_neg == 0 or num_nodes < 2:
        return torch.empty((2, 0), dtype=torch.long, device=pos_edge_index.device)
        
    # Simple random sampling (may rarely sample a true positive, but fast)
    neg_i = torch.randint(0, num_nodes, (num_neg,), device=pos_edge_index.device)
    neg_j = torch.randint(0, num_nodes, (num_neg,), device=pos_edge_index.device)
    
    return torch.stack([neg_i, neg_j], dim=0)

def train_vgae(epochs=50, batch_size=8, output_dir="./hgnn_checkpoints/vgae_pretrained"):
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    converter = PublicDatasetGraphConverter()
    all_graphs = []
    
    # Load dataset (using UNSW as example)
    logger.info("Loading UNSW-NB15 for VGAE pre-training...")
    dataset_path = Path("datasets/unsw_nb15/mitre_format.csv")
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return
        
    df = pd.read_csv(dataset_path)
    
    # Group into mini-graphs
    for campaign_id, campaign_df in df.groupby('campaign_id'):
        if len(campaign_df) < 10:
            continue
        for chunk_start in range(0, len(campaign_df), 500):
            chunk_df = campaign_df.iloc[chunk_start:chunk_start + 500]
            graph = converter.convert(chunk_df)
            if graph is not None and 'alert' in graph.node_types:
                all_graphs.append(graph.to(device))
                
    logger.info(f"Loaded {len(all_graphs)} graphs to {device}")
    
    # Initialize Backbone and VGAE wrapper
    backbone = MITREHeteroGNN(
        alert_feature_dim=15,
        hidden_dim=128,
        num_layers=1,
        num_clusters=10
    ).to(device)
    
    vgae = HeteroVGAE(encoder=backbone, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(vgae.parameters(), lr=1e-3, weight_decay=1e-5)
    
    output_path = Path(output_dir) / "vgae_best.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        vgae.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        n_steps = 0
        
        for i in range(0, len(all_graphs), batch_size):
            batch_graphs = all_graphs[i:i+batch_size]
            optimizer.zero_grad()
            
            batch_loss = 0
            for graph in batch_graphs:
                pos_edges = get_positive_edges(graph)
                n_alerts = graph['alert'].x.size(0)
                
                if pos_edges.size(1) == 0:
                    continue
                    
                neg_edges = sample_negative_edges(pos_edges, n_alerts)
                
                # Forward pass
                mu, logvar = vgae.encode(graph)
                z = vgae.reparameterize(mu, logvar)
                
                # Decode positive and negative edges
                pos_logits = vgae.decode(z, pos_edges)
                neg_logits = vgae.decode(z, neg_edges)
                
                # Binary Cross Entropy
                pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
                neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
                recon_loss = pos_loss + neg_loss
                
                # KL Divergence
                kl_loss = vgae.kl_loss(mu, logvar)
                
                # Total loss (ELBO)
                loss = recon_loss + (1.0 / n_alerts) * kl_loss
                batch_loss += loss
                
                total_recon += recon_loss.item()
                total_kl += kl_loss.item()
                total_loss += loss.item()
                n_steps += 1
                
            if batch_loss > 0:
                batch_loss.backward()
                optimizer.step()
                
        avg_loss = total_loss / max(n_steps, 1)
        logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} (Recon={total_recon/max(n_steps,1):.4f}, KL={total_kl/max(n_steps,1):.4f})")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save the backbone weights (not the VGAE wrapper)
            torch.save({
                'epoch': epoch,
                'model_state_dict': backbone.state_dict(),
                'loss': best_loss,
                'hidden_dim': 128,
            }, output_path)
            
    logger.info(f"VGAE pre-training complete. Backbone saved to {output_path}")

if __name__ == "__main__":
    train_vgae()
