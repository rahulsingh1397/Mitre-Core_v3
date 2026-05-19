import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import List, Tuple, Dict, Optional
import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hgnn.hgnn_correlation import MITREHeteroGNN, AlertToGraphConverter
from utils.clustering import evaluate_clustering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SupervisedPrototypeHead(nn.Module):
    """
    Learnable class prototypes for supervised HGNN training.
    Replaces HDBSCAN density clustering with explicit centroid distance.
    """
    def __init__(self, hidden_dim: int, num_classes: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        # Learnable prototypes C in R^{K x d}
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.prototypes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits based on cosine similarity to prototypes.
        Args:
            x: [N, hidden_dim] embeddings
        Returns:
            logits: [N, num_classes]
        """
        # L2 normalize embeddings and prototypes
        x_norm = F.normalize(x, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        # Cosine similarity scaled by a temperature factor (learned or fixed)
        # Using a fixed tau=0.1 is common for prototype learning
        tau = 0.1
        logits = torch.matmul(x_norm, proto_norm.t()) / tau
        return logits

def train_supervised_prototypes(
    df: pd.DataFrame, 
    label_col: str = "campaign_id",
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = 1,
    epochs: int = 100,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: str = "models/checkpoints/hgnn_prototypes"
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter out unlabeled/unknown data
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col} not found in dataframe")
        
    # Map labels to 0...K-1
    unique_labels = sorted([l for l in df[label_col].unique() if pd.notna(l) and str(l).lower() != 'unknown'])
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)
    logger.info(f"Found {num_classes} valid classes for prototype learning")
    
    if "AlertId" not in df.columns:
        df["AlertId"] = [f"alert_{i}" for i in range(len(df))]

    # Convert to graph
    converter = AlertToGraphConverter()
    data = converter.convert(df)
    data = data.to(device)
    
    # Create target tensor
    alert_ids = df["AlertId"].values
    targets = torch.zeros(len(df), dtype=torch.long, device=device)
    valid_mask = torch.zeros(len(df), dtype=torch.bool, device=device)
    
    for i, aid in enumerate(alert_ids):
        l = df.iloc[i][label_col]
        if pd.notna(l) and str(l).lower() != 'unknown' and l in label_to_idx:
            targets[i] = label_to_idx[l]
            valid_mask[i] = True
            
    logger.info(f"Using {valid_mask.sum().item()} labeled alerts out of {len(df)} total")
    
    # Initialize model
    hgnn = MITREHeteroGNN(
        hidden_dim=hidden_dim, 
        num_heads=num_heads, 
        num_layers=num_layers
    ).to(device)
    
    prototype_head = SupervisedPrototypeHead(hidden_dim, num_classes).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': hgnn.parameters(), 'lr': lr},
        {'params': prototype_head.parameters(), 'lr': lr * 10} # Fast prototype learning
    ])
    
    criterion = nn.CrossEntropyLoss()
    
    hgnn.train()
    prototype_head.train()
    
    best_ari = -1.0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        _, x_dict = hgnn(data)
        alert_emb = x_dict["alert"]
        
        # Compute logits for valid alerts
        valid_emb = alert_emb[valid_mask]
        valid_targets = targets[valid_mask]
        
        logits = prototype_head(valid_emb)
        loss = criterion(logits, valid_targets)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            hgnn.eval()
            prototype_head.eval()
            with torch.no_grad():
                _, x_dict_val = hgnn(data)
                all_logits = prototype_head(x_dict_val["alert"])
                preds = torch.argmax(all_logits, dim=1).cpu().numpy()
                
                # Evaluate only on valid subset
                valid_preds = preds[valid_mask.cpu().numpy()]
                valid_true = targets[valid_mask].cpu().numpy()
                
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(valid_true, valid_preds)
                
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Train ARI: {ari:.4f}")
                
                if ari > best_ari:
                    best_ari = ari
                    torch.save({
                        'hgnn_state_dict': hgnn.state_dict(),
                        'prototype_state_dict': prototype_head.state_dict(),
                        'label_to_idx': label_to_idx,
                        'num_classes': num_classes,
                        'ari': ari
                    }, Path(output_dir) / "best_prototype_model.pt")
                    
            hgnn.train()
            prototype_head.train()
            
    logger.info(f"Training complete. Best Train ARI: {best_ari:.4f}")
    return hgnn, prototype_head

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset parquet/csv")
    parser.add_argument("--label_col", type=str, default="campaign_id", help="Label column for supervised training")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="hgnn_checkpoints/prototypes")
    args = parser.parse_args()
    
    if args.dataset.endswith('.parquet'):
        df = pd.read_parquet(args.dataset)
    else:
        df = pd.read_csv(args.dataset)
        
    train_supervised_prototypes(df, label_col=args.label_col, epochs=args.epochs, output_dir=args.output_dir)
