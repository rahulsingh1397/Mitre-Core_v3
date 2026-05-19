# EXPERIMENTAL: This file contains early research explorations (VGAE) and is not part of the production pipeline.
"""
Train Link Prediction (VGAE) on MITRE-CORE

Implements Unsupervised Link Prediction (Graph Autoencoder) as a pre-training
objective for the MITRE-CORE HGNN. This is a robust alternative to NT-Xent
that uses the inherent graph structure (shared IPs, hosts, users) as the
supervision signal.
"""

import os
import sys
import random
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hgnn.hgnn_correlation import MITREHeteroGNN
from training.train_on_datasets import DatasetTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mitre-core.train_link_pred")


class CampaignLinkPredictor(nn.Module):
    """
    Link prediction head for unsupervised GNN training.
    Predicts if two alerts should be connected based on their embeddings.
    """
    def __init__(self, gnn: MITREHeteroGNN, hidden_dim: int = 128):
        super().__init__()
        self.encoder = gnn
        self.link_pred = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(self, graph, pos_pairs, neg_pairs):
        """
        Compute binary cross-entropy loss for link prediction.
        pos_pairs: [2, num_pos] indices of true edges
        neg_pairs: [2, num_neg] indices of false edges
        """
        # Get node embeddings
        _, x_dict = self.encoder(graph)
        embs = x_dict['alert']
        
        # Positive pairs
        pos_src = embs[pos_pairs[0]]
        pos_dst = embs[pos_pairs[1]]
        pos_scores = self.link_pred(pos_src, pos_dst).squeeze(-1)
        
        # Negative pairs
        neg_src = embs[neg_pairs[0]]
        neg_dst = embs[neg_pairs[1]]
        neg_scores = self.link_pred(neg_src, neg_dst).squeeze(-1)
        
        # Combine and compute loss
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones_like(pos_scores),
            torch.zeros_like(neg_scores)
        ])
        
        return F.binary_cross_entropy_with_logits(scores, labels)


def extract_positive_pairs(graph) -> torch.Tensor:
    """Extract all alert-to-alert edges as positive pairs."""
    pos_pairs = []
    
    for edge_type in [
        ("alert", "shares_ip", "alert"),
        ("alert", "shares_host", "alert"),
        ("alert", "temporal_near", "alert")
    ]:
        if edge_type in graph.edge_types:
            ei = graph[edge_type].edge_index
            if ei.numel() > 0:
                pos_pairs.append(ei)
                
    if not pos_pairs:
        return torch.empty((2, 0), dtype=torch.long, device=graph['alert'].x.device)
        
    return torch.cat(pos_pairs, dim=1)


def sample_negative_pairs(num_nodes: int, num_samples: int, device: torch.device) -> torch.Tensor:
    """Sample random node pairs as negatives."""
    if num_nodes < 2 or num_samples <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
        
    src = torch.randint(0, num_nodes, (num_samples,), device=device)
    dst = torch.randint(0, num_nodes, (num_samples,), device=device)
    
    # Avoid self-loops
    mask = src != dst
    return torch.stack([src[mask], dst[mask]], dim=0)


def train_link_prediction(
    dataset_name: str,
    epochs: int = 50,
    hidden_dim: int = 128,
    output_dir: str = "./hgnn_checkpoints/link_pred"
):
    """Train HGNN using unsupervised link prediction."""
    trainer = DatasetTrainer(output_path=output_dir)
    device = trainer.device
    
    logger.info(f"Loading {dataset_name}...")
    df = trainer.load_mitre_dataset(dataset_name)
    if df is None:
        return
        
    train_df, _, _, _ = trainer.prepare_training_data(df)
    
    from training.train_on_datasets import PublicDatasetGraphConverter
    converter = PublicDatasetGraphConverter()
    
    # Create mini-campaign graphs
    campaign_size = 50
    num_campaigns = len(train_df) // campaign_size
    train_graphs = []
    
    logger.info(f"Creating {num_campaigns} mini-graphs for training...")
    for i in range(0, len(train_df), campaign_size):
        mini_df = train_df.iloc[i:i+campaign_size]
        graph = converter.convert(mini_df)
        if graph is not None and 'alert' in graph.node_types:
            train_graphs.append(graph)
            
    train_graphs = trainer._ensure_consistent_node_types(train_graphs)
    if not train_graphs:
        logger.error("No valid training graphs created.")
        return
        
    alert_feature_dim = train_graphs[0]['alert'].x.shape[1]
    
    # Initialize model
    hgnn = MITREHeteroGNN(
        alert_feature_dim=alert_feature_dim,
        hidden_dim=hidden_dim,
        num_clusters=10 # Dummy value for pre-training
    ).to(device)
    
    predictor = CampaignLinkPredictor(hgnn, hidden_dim).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.001)
    
    logger.info(f"Starting Link Prediction Pre-training ({epochs} epochs)")
    best_loss = float('inf')
    output_path = Path(output_dir) / f"{dataset_name}_best_link_pred.pt"
    
    for epoch in range(epochs):
        predictor.train()
        total_loss = 0
        n_steps = 0
        
        random.shuffle(train_graphs)
        
        for graph in train_graphs:
            graph = graph.to(device)
            num_alerts = graph['alert'].num_nodes
            
            pos_pairs = extract_positive_pairs(graph)
            if pos_pairs.numel() == 0:
                continue
                
            # Sample equal number of negatives
            neg_pairs = sample_negative_pairs(num_alerts, pos_pairs.size(1), device)
            if neg_pairs.numel() == 0:
                continue
                
            optimizer.zero_grad()
            loss = predictor(graph, pos_pairs, neg_pairs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_steps += 1
            
        avg_loss = total_loss / max(n_steps, 1)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': hgnn.state_dict(),
                'predictor_state_dict': predictor.state_dict(),
                'loss': best_loss,
                'hidden_dim': hidden_dim,
            }, output_path)
            
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
    logger.info(f"Pre-training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, nargs='+', default=['unsw_nb15'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='./hgnn_checkpoints/link_pred')
    args = parser.parse_args()
    
    for ds in args.datasets:
        train_link_prediction(ds, epochs=args.epochs, output_dir=args.output_dir)
