#!/usr/bin/env python3
"""
DeepCluster GNN Baseline for MITRE-CORE

Implements DeepCluster (Caron et al., 2018) with frozen HGNN backbone:
1. Extract backbone embeddings from network_v9_v3
2. Iterative K-Means clustering + pseudo-label assignment
3. Train linear classifier on pseudo-labels (cross-entropy)
4. Final K-Means clustering for evaluation

This provides a self-supervised GNN baseline to compare against MITRE-CORE.
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from hgnn.hgnn_correlation import HGNNCorrelationEngine
from training.train_on_datasets import PublicDatasetGraphConverter, apply_edge_dropout

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DeepClusterTrainer:
    """
    DeepCluster with frozen HGNN backbone.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        n_clusters: int,
        device: str = 'cuda',
        batch_size: int = 256,
        n_epochs: int = 30,
        clustering_interval: int = 5,
        temperature: float = 0.07,
    ):
        self.checkpoint_path = checkpoint_path
        self.n_clusters = n_clusters
        self.device = device
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clustering_interval = clustering_interval
        self.temperature = temperature
        
        # Initialize components
        self.correlation_engine = None
        self.linear_head = None
        self.converter = PublicDatasetGraphConverter()
        
    def setup(self):
        """Initialize HGNN engine and linear classifier."""
        logger.info(f"Loading HGNN from {self.checkpoint_path}")
        
        self.correlation_engine = HGNNCorrelationEngine(
            model_path=self.checkpoint_path,
            device=self.device,
            use_geometric_confidence=False,  # Don't use clustering during training
            pure_unsupervised=False,  # We'll use our own clustering
        )
        
        # Linear classifier on top of frozen embeddings
        hidden_dim = 128  # network_v9_v3 hidden dimension
        self.linear_head = nn.Linear(hidden_dim, self.n_clusters).to(self.device)
        
        # Freeze HGNN backbone
        for param in self.correlation_engine.model.parameters():
            param.requires_grad = False
        
        logger.info(f"DeepCluster setup complete: {self.n_clusters} clusters, {self.n_epochs} epochs")
    
    def extract_embeddings_batched(self, df: pd.DataFrame) -> torch.Tensor:
        """Extract embeddings in batches to prevent OOM."""
        embeddings_list = []
        
        # Process in chunks of 1000 alerts
        chunk_size = 1000
        for i in range(0, len(df), chunk_size):
            chunk_df = df.iloc[i:i+chunk_size].reset_index(drop=True)
            
            try:
                # Extract embeddings using correlation engine
                chunk_embeddings = self.correlation_engine.extract_embeddings(chunk_df)
                if chunk_embeddings is not None:
                    embeddings_list.append(chunk_embeddings)
                    logger.debug(f"Successfully extracted embeddings for chunk {i}: {chunk_embeddings.shape}")
                else:
                    logger.warning(f"Got None embeddings for chunk {i}")
            except Exception as e:
                logger.warning(f"Failed to extract embeddings for chunk {i}: {e}")
                continue
        
        if not embeddings_list:
            raise RuntimeError("Failed to extract any embeddings")
        
        # Concatenate all embeddings (convert numpy to tensor if needed)
        if embeddings_list:
            if isinstance(embeddings_list[0], np.ndarray):
                all_embeddings = torch.from_numpy(np.concatenate(embeddings_list, axis=0))
            else:
                all_embeddings = torch.cat(embeddings_list, dim=0)
        else:
            raise RuntimeError("No embeddings extracted")
        
        logger.info(f"Extracted embeddings: {all_embeddings.shape}")
        
        return all_embeddings
    
    def perform_clustering(self, embeddings: torch.Tensor) -> np.ndarray:
        """Perform K-Means clustering on embeddings."""
        # Convert to numpy
        emb_np = embeddings.detach().cpu().numpy()
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        pseudo_labels = kmeans.fit_predict(emb_np)
        
        logger.info(f"K-Means clustering: {len(np.unique(pseudo_labels))} clusters found")
        
        return pseudo_labels
    
    def train_linear_head(self, embeddings: torch.Tensor, labels: np.ndarray) -> float:
        """Train linear classifier on pseudo-labels."""
        # Create dataset
        dataset = TensorDataset(embeddings, torch.tensor(labels, dtype=torch.long))
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer (only linear head parameters)
        optimizer = torch.optim.Adam(self.linear_head.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.linear_head.train()
        total_loss = 0.0
        n_batches = 0
        
        for epoch in range(5):  # Train for 5 epochs on current pseudo-labels
            epoch_loss = 0.0
            for batch_emb, batch_labels in dataloader:
                batch_emb = batch_emb.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Forward pass
                logits = self.linear_head(batch_emb)
                loss = criterion(logits, batch_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            total_loss += epoch_loss
        
        avg_loss = total_loss / (n_batches * 5)
        logger.info(f"Linear head training loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def run_deepcluster(self, df: pd.DataFrame, true_labels: np.ndarray) -> Dict[str, float]:
        """
        Run DeepCluster algorithm.
        
        Returns:
            Dict with ARI, NMI, and final pseudo-labels
        """
        logger.info(f"Running DeepCluster on {len(df)} samples")
        
        # Extract embeddings
        embeddings = self.extract_embeddings_batched(df)
        
        # Initial clustering
        pseudo_labels = self.perform_clustering(embeddings)
        
        # Iterative training
        for epoch in range(self.n_epochs):
            # Re-cluster every clustering_interval epochs
            if epoch % self.clustering_interval == 0:
                pseudo_labels = self.perform_clustering(embeddings)
            
            # Train linear head on current pseudo-labels
            loss = self.train_linear_head(embeddings, pseudo_labels)
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}: loss={loss:.4f}")
        
        # Final clustering for evaluation
        final_pseudo_labels = self.perform_clustering(embeddings)
        
        # Evaluate against true labels
        ari = adjusted_rand_score(true_labels, final_pseudo_labels)
        nmi = normalized_mutual_info_score(true_labels, final_pseudo_labels)
        
        logger.info(f"DeepCluster final: ARI={ari:.4f}, NMI={nmi:.4f}")
        
        return {
            'ari': ari,
            'nmi': nmi,
            'pseudo_labels': final_pseudo_labels,
            'n_clusters_found': len(np.unique(final_pseudo_labels)),
        }


def load_dataset(dataset_name: str, config: dict) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """Load and preprocess dataset for DeepCluster."""
    logger.info(f"Loading {dataset_name} dataset...")
    
    # Load data
    data_path = Path(config["path"])
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    df = pd.read_parquet(data_path) if data_path.suffix == ".parquet" else pd.read_csv(data_path)
    
    # Sample to prevent OOM (same as baseline script)
    SAMPLE_SIZE = 10000
    if len(df) > SAMPLE_SIZE:
        logger.info(f"Sampling {SAMPLE_SIZE} records from {len(df)}")
        df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    
    # Extract labels
    label_col = config["label_col"]
    labels = LabelEncoder().fit_transform(df[label_col].fillna("UNKNOWN").astype(str))
    n_true_classes = config["true_clusters"]
    
    logger.info(f"Loaded {dataset_name}: {len(df)} samples, {n_true_classes} true classes")
    
    return df, labels, n_true_classes


def get_dataset_config(dataset_name: str) -> dict:
    """Get dataset configuration matching baseline script."""
    configs = {
        "UNSW-NB15": {
            "path": "datasets/unsw_nb15/mitre_format.csv",
            "label_col": "campaign_id",
            "true_clusters": 8,
        },
        "NSL-KDD": {
            "path": "datasets/nsl_kdd/mitre_format.csv",
            "label_col": "tactic",
            "true_clusters": 10,
        },
        "CICIDS2017": {
            "path": "datasets/cicids2017/mitre_format.parquet",
            "label_col": "campaign_id",
            "true_clusters": 16,
        },
        "TON_IoT": {
            "path": "datasets/TON_IoT/mitre_format.parquet",
            "label_col": "campaign_id",
            "true_clusters": 10,
        },
        "SQTK_SIEM_kcluster": {
            "path": "datasets/SQTK_SIEM/mitre_core_format.csv",
            "label_col": "kcluster",
            "true_clusters": 11,
        },
        "OpTC": {
            "path": "datasets/DARPA_OpTC/processed_optc_full.csv",
            "label_col": "CampaignId",
            "true_clusters": 2,
        },
    }
    
    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return configs[dataset_name]


def main():
    parser = argparse.ArgumentParser(description="DeepCluster GNN baseline")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["UNSW-NB15", "NSL-KDD", "CICIDS2017", "TON_IoT", "SQTK_SIEM_kcluster", "OpTC"],
                       help="Dataset to run DeepCluster on")
    parser.add_argument("--checkpoint", type=str, 
                       default="hgnn_checkpoints/network_v9_v3/network_it_best.pt",
                       help="HGNN checkpoint path")
    parser.add_argument("--output", type=str, default="experiments/results/deepcluster_results.csv",
                       help="Output CSV file")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    
    args = parser.parse_args()
    
    # Load dataset
    config = get_dataset_config(args.dataset)
    df, true_labels, n_true_classes = load_dataset(args.dataset, config)
    
    # Initialize DeepCluster
    trainer = DeepClusterTrainer(
        checkpoint_path=args.checkpoint,
        n_clusters=n_true_classes,
        device=args.device,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    # Setup
    trainer.setup()
    
    # Run DeepCluster
    start_time = time.time()
    results = trainer.run_deepcluster(df, true_labels)
    runtime = time.time() - start_time
    
    # Prepare results
    result_row = {
        "dataset": args.dataset,
        "method": "DeepCluster",
        "runtime_seconds": runtime,
        "n_true_classes": n_true_classes,
        "ari": results["ari"],
        "nmi": results["nmi"],
        "n_pred_clusters": results["n_clusters_found"],
        "binary_ari": float("nan"),  # Not applicable for multi-class
        "coverage": 1.0,  # K-Means covers all points
    }
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Append to CSV or create new file
    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        result_df = pd.concat([existing_df, pd.DataFrame([result_row])], ignore_index=True)
    else:
        result_df = pd.DataFrame([result_row])
    
    result_df.to_csv(output_path, index=False)
    
    logger.info(f"Results saved to {output_path}")
    logger.info(f"DeepCluster on {args.dataset}: ARI={results['ari']:.4f}, NMI={results['nmi']:.4f}, runtime={runtime:.1f}s")


if __name__ == "__main__":
    main()
