#!/usr/bin/env python3
# EXPERIMENTAL: v10 hybrid training. Caused UNSW-NB15 regression (<0.02 ARI).
# Not in canonical pipeline. See MEMORY.md v2.22 WEEK4 section.
# EXPERIMENTAL: v10 hybrid training. Caused UNSW regression. See MEMORY.md v2.22 WEEK4.
"""
Hybrid Adaptive Training for Cross-Schema Campaign Clustering (v10)

Implements the ECML-PKDD/KDD Applied contribution:
1. Campaign-Temporal Contrastive loss (unsupervised, always on)
2. Per-sample weighted SupCon (supervised, w_i in {1.0, 0.7})
3. EM pseudo-label refresh (confidence > 0.8, w=0.3)

Combines #1 + #2 + #3 from the refined research plan.
"""

import os
import sys
import time
import logging
import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from hgnn.hgnn_correlation import MITREHeteroGNN
from hgnn.contrastive_loss import NTXentLoss
from hgnn.supcon_loss import SupConLoss
from training.train_on_datasets import PublicDatasetGraphConverter, apply_edge_dropout
from training.train_on_datasets import apply_combined_augmentation
from utils.clustering import hdbscan_cluster_with_confidence
from utils.metrics import adjusted_rand_score, normalized_mutual_info_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridAdaptiveTrainer:
    """
    Hybrid adaptive trainer with per-sample confidence weighting.
    
    Loss formulation:
    L_total = L_temporal_contrastive (always on)
            + w_sup * L_sup_contrastive (per-sample weighted)
            + w_pseudo * L_pseudo_contrastive (EM-refreshed)
    """
    
    def __init__(
        self,
        datasets: Dict[str, pd.DataFrame],
        label_metadata: pd.DataFrame,
        hidden_dim: int = 128,
        temperature: float = 0.07,
        sup_temperature: float = 0.07,
        device: str = 'cuda',
        temporal_window_hours: float = 2.0,
        pseudo_confidence_threshold: float = 0.8,
        pseudo_refresh_every_k_epochs: int = 5,
        lambda_sup: float = 1.0,
        lambda_pseudo: float = 0.5,
    ):
        self.datasets = datasets
        self.label_metadata = label_metadata
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.sup_temperature = sup_temperature
        self.device = device
        self.temporal_window_hours = temporal_window_hours
        self.pseudo_confidence_threshold = pseudo_confidence_threshold
        self.pseudo_refresh_every_k_epochs = pseudo_refresh_every_k_epochs
        self.lambda_sup = lambda_sup
        self.lambda_pseudo = lambda_pseudo
        
        # Initialize components
        self.converter = PublicDatasetGraphConverter()
        self.temporal_loss = NTXentLoss(temperature=temperature)
        self.supcon_loss = SupConLoss(temperature=sup_temperature)
        
        # Will be initialized in setup_model()
        self.hgnn = None
        self.pseudo_labels = {}  # dataset_name -> pseudo_label_array
        
    def setup_model(self, base_checkpoint: str):
        """Load base checkpoint and setup model."""
        logger.info(f"Loading base checkpoint: {base_checkpoint}")
        
        ckpt = torch.load(base_checkpoint, map_location=self.device, weights_only=False)
        hidden_dim = ckpt.get('hidden_dim', 128)
        num_clusters = ckpt.get('num_clusters', 10)
        vocab_sizes = ckpt.get('vocab_sizes', None)
        
        self.hgnn = MITREHeteroGNN(
            alert_feature_dim=6,  # B1: 6-dim base features
            hidden_dim=hidden_dim,
            num_layers=1,
            num_clusters=num_clusters,
            vocab_sizes=vocab_sizes,
        ).to(self.device)
        
        # Materialise lazy modules BEFORE loading checkpoint
        logger.info("Materialising lazy modules...")
        # Create a minimal dummy graph to trigger lazy module materialisation
        from training.train_on_datasets import PublicDatasetGraphConverter
        dummy_df = pd.DataFrame({
            'timestamp': [pd.Timestamp('2024-01-01'), pd.Timestamp('2024-01-01 00:05:00')],
            'src_ip': ['10.0.0.1', '10.0.0.2'],
            'dst_ip': ['10.0.0.2', '10.0.0.3'],
            'src_port': [80, 443],
            'dst_port': [1234, 8080],
            'protocol': [6, 6],
            'alert_type': [1, 2],
            'severity': [1, 2],
            'category': [1, 2],
            'src_hostname': ['host1', 'host2'],
            'dst_hostname': ['host2', 'host3'],
            'src_user': ['user1', 'user2'],
            'dst_user': ['user2', 'user3'],
        })
        dummy_converter = PublicDatasetGraphConverter()
        dummy_graph = dummy_converter.convert(dummy_df)
        if dummy_graph is not None:
            dummy_graph = dummy_graph.to(self.device)
            with torch.no_grad():
                _ = self.hgnn(dummy_graph)
            logger.info("Lazy modules materialised successfully")
        
        # Load checkpoint
        missing_keys, unexpected_keys = self.hgnn.load_state_dict(ckpt['model_state_dict'], strict=False)
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        # Check if alert_raw_proj is in missing keys (should not be after materialisation)
        if any('alert_raw_proj' in key for key in missing_keys):
            logger.error("alert_raw_proj still missing after materialisation - this will cause NaN!")
        
        # Setup losses
        self.temporal_loss = NTXentLoss(temperature=self.temperature)
        sup_temperature = 0.07
        self.supcon_loss = SupConLoss(temperature=sup_temperature)
        
        logger.info(f"Model loaded: hidden_dim={hidden_dim}, num_clusters={num_clusters}")
        
    def get_sample_weights(self, dataset_name: str, indices: List[int]) -> torch.Tensor:
        """
        Get per-sample confidence weights for hybrid loss.
        
        Returns:
            Tensor of shape (N,) with weights in {1.0, 0.7, 0.3, 0.0}
        """
        # Filter label_metadata for this dataset and indices
        meta = self.label_metadata[
            (self.label_metadata['dataset'] == dataset_name) & 
            (self.label_metadata['sample_idx'].isin(indices))
        ]
        
        # Create weight mapping
        weight_map = dict(zip(meta['sample_idx'], meta['weight']))
        
        weights = []
        for idx in indices:
            weight = weight_map.get(idx, 0.0)  # Default to 0.0 if not found
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32, device=self.device)
    
    def build_temporal_positive_pairs(self, embeddings: torch.Tensor, timestamps: np.ndarray) -> List[Tuple[int, int]]:
        """
        Build positive pairs based on temporal proximity within window.
        
        Args:
            embeddings: Tensor of shape (N, D)
            timestamps: Array of shape (N,) with Unix timestamps
            
        Returns:
            List of (i, j) pairs where |t_i - t_j| <= temporal_window_hours
        """
        positive_pairs = []
        window_seconds = self.temporal_window_hours * 3600
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if abs(timestamps[i] - timestamps[j]) <= window_seconds:
                    positive_pairs.append((i, j))
        
        return positive_pairs
    
    def compute_temporal_contrastive_loss(self, embeddings: torch.Tensor, timestamps: np.ndarray, campaign_labels: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Compute temporal contrastive loss using vectorised operations.
        
        Args:
            embeddings: Tensor of shape (N, D) with alert embeddings
            timestamps: Array of shape (N,) with Unix timestamps
            campaign_labels: Optional array of shape (N,) with campaign IDs for cross-campaign guard
            
        Returns:
            Temporal NT-Xent loss scalar
        """
        if len(timestamps) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Convert timestamps to tensor on the same device
        t = torch.as_tensor(timestamps, dtype=torch.float32, device=embeddings.device)
        
        # Build temporal positive pairs using vectorised operations
        window_seconds = self.temporal_window_hours * 3600
        dt = (t[:, None] - t[None, :]).abs()  # Pairwise time differences
        pos_mask = (dt <= window_seconds) & ~torch.eye(len(t), dtype=torch.bool, device=self.device)
        
        # Guard against cross-campaign positives when campaign_labels available
        if campaign_labels is not None:
            labels_tensor = torch.as_tensor(campaign_labels, device=embeddings.device)
            campaign_mask = (labels_tensor[:, None] == labels_tensor[None, :])
            pos_mask = pos_mask & campaign_mask
        
        # Get positive pairs list for NTXentLoss
        positive_pairs = [(i, j) for i in range(len(t)) for j in range(i+1, len(t)) if pos_mask[i, j]]
        
        if len(positive_pairs) == 0:
            return torch.tensor(0.0, device=self.device)
        
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=-1)
        
        # Use NTXent loss with temporal positives
        loss = self.temporal_loss(embeddings_norm, positive_pairs)
        
        return loss
    
    def refresh_pseudo_labels(self, dataset_name: str, df_sample: pd.DataFrame, embeddings: torch.Tensor):
        """
        Refresh pseudo-labels using HDBSCAN on current embeddings and update dataframe.
        """
        if embeddings.shape[0] < 5:  # Require minimum samples for HDBSCAN
            logger.warning(f"Not enough embeddings ({embeddings.shape[0]}) to run HDBSCAN for {dataset_name}")
            return
            
        # Convert to numpy
        emb_np = embeddings.detach().cpu().numpy()
        
        # Check for NaN values in embeddings
        if np.isnan(emb_np).any():
            logger.warning(f"NaN values found in embeddings for {dataset_name}, skipping pseudo-label refresh")
            return
            
        # HDBSCAN clustering
        try:
            cluster_labels, confidences = hdbscan_cluster_with_confidence(
                emb_np, min_cluster_size=5, min_samples=3
            )
        except ValueError as e:
            logger.error(f"HDBSCAN failed for {dataset_name}: {e}")
            return
            
        # Initialize columns if they don't exist
        if 'pseudo_label' not in self.datasets[dataset_name].columns:
            self.datasets[dataset_name]['pseudo_label'] = -1
            self.datasets[dataset_name]['pseudo_confidence'] = 0.0
            
        # Update the sampled rows in the main dataframe
        self.datasets[dataset_name].loc[df_sample.index, 'pseudo_label'] = cluster_labels
        self.datasets[dataset_name].loc[df_sample.index, 'pseudo_confidence'] = confidences
        
        # Log statistics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        avg_confidence = np.mean(confidences[confidences > 0]) if (confidences > 0).any() else 0.0
        logger.info(f"Pseudo-labels refreshed for {dataset_name}: {n_clusters} clusters, avg_conf={avg_confidence:.3f}")
    
    def compute_pseudo_contrastive_loss(self, chunk: pd.DataFrame, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss on high-confidence pseudo-labels for the current batch.
        """
        if 'pseudo_label' not in chunk.columns:
            return torch.tensor(0.0, device=self.device)
            
        labels = chunk['pseudo_label'].values
        confidences = chunk['pseudo_confidence'].values
        
        # Filter high-confidence samples that are not noise (-1)
        high_conf_mask = (confidences >= self.pseudo_confidence_threshold) & (labels != -1)
        if high_conf_mask.sum() < 4:  # Need at least 4 samples for meaningful contrastive loss
            return torch.tensor(0.0, device=self.device)
        
        high_conf_indices = np.where(high_conf_mask)[0]
        high_conf_labels = labels[high_conf_mask]
        high_conf_embeddings = embeddings[high_conf_indices]
        
        # Remap pseudo-labels to contiguous range [0..K-1] to prevent CUDA out-of-bounds
        unique_labels, remapped_labels = torch.unique(
            torch.tensor(high_conf_labels, device=self.device),
            return_inverse=True
        )
        
        # Normalize embeddings
        embeddings_norm = F.normalize(high_conf_embeddings, dim=-1)
        
        # Use SupCon loss with remapped pseudo-labels
        loss = self.supcon_loss(embeddings_norm, remapped_labels)
        
        return loss * self.lambda_pseudo
    
    def train_epoch(self, epoch: int, batch_size: int = 4) -> Dict[str, float]:
        """
        Train one epoch across all datasets.
        """
        self.hgnn.train()
        epoch_losses = defaultdict(float)
        dataset_losses = defaultdict(lambda: defaultdict(float))  # Track per-dataset losses
        n_batches = 0
        
        # Refresh pseudo-labels if needed
        if epoch % self.pseudo_refresh_every_k_epochs == 0:
            logger.info(f"Refreshing pseudo-labels at epoch {epoch}")
            for dataset_name in self.datasets.keys():
                # Sample a subset for pseudo-label computation (to save time)
                df = self.datasets[dataset_name].sample(n=min(5000, len(self.datasets[dataset_name])))
                graph = self.converter.convert(df)
                if graph is None or 'alert' not in graph.node_types:
                    continue
                
                graph = graph.to(self.device)
                _, embeddings = self.hgnn(graph)
                if 'alert' in embeddings:
                    self.refresh_pseudo_labels(dataset_name, df, embeddings['alert'])
        
        # Train on each dataset
        for dataset_name, df in self.datasets.items():
            # Sample campaigns (if dataset has campaign_id)
            if 'campaign_id' in df.columns:
                campaign_groups = [(cid, grp) for cid, grp in df.groupby('campaign_id') if len(grp) >= 4]
                random.shuffle(campaign_groups)
                
                for batch_start in range(0, len(campaign_groups), batch_size):
                    batch_campaigns = campaign_groups[batch_start:batch_start + batch_size]
                    if not batch_campaigns:
                        continue
                    
                    # Process batch
                    batch_loss = self._process_batch(dataset_name, batch_campaigns, epoch_losses, dataset_losses)
                    if batch_loss > 0:
                        n_batches += 1
            else:
                # For datasets without campaign_id, sample random batches
                for _ in range(10):  # 10 random batches per dataset
                    batch_df = df.sample(n=min(1000, len(df)))
                    graph = self.converter.convert(batch_df)
                    if graph is None or 'alert' not in graph.node_types:
                        continue
                    
                    graph = graph.to(self.device)
                    _, embeddings = self.hgnn(graph)
                    if 'alert' not in embeddings:
                        continue
                    
                    # Temporal contrastive loss only (no supervised labels)
                    if 'timestamp' in batch_df.columns:
                        timestamps = batch_df['timestamp'].values.astype(np.int64) // 10**9
                        
                        # Ensure sizes match
                        min_len = min(len(timestamps), embeddings['alert'].size(0))
                        timestamps = timestamps[:min_len]
                        emb_alert_temp = embeddings['alert'][:min_len]
                        
                        # No campaign labels available for unsupervised datasets - use narrower window
                        temp_loss = self.compute_temporal_contrastive_loss(emb_alert_temp, timestamps, campaign_labels=None)
                        
                        self.optimizer.zero_grad()
                        if isinstance(temp_loss, torch.Tensor) and temp_loss.requires_grad:
                            temp_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.hgnn.parameters(), 1.0)
                            self.optimizer.step()
                        
                        val = temp_loss.item() if isinstance(temp_loss, torch.Tensor) else temp_loss
                        epoch_losses['temporal'] += val
                        dataset_losses[dataset_name]['temporal'] += val
                        n_batches += 1
                        
        if n_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= n_batches
                
        return epoch_losses, dict(dataset_losses)
    
    def _process_batch(self, dataset_name: str, batch_campaigns: List[Tuple], epoch_losses: Dict[str, float], dataset_losses: Dict[str, Dict[str, float]]) -> float:
        """
        Process a batch of campaigns and compute hybrid loss.
        """
        batch_campaign_losses = []
        
        for label_int, grp in batch_campaigns:
            if len(grp) < 4:
                continue
            
            # Sample chunk to prevent OOM
            max_chunk = 400
            if len(grp) > max_chunk:
                chunk = grp.sample(n=max_chunk).reset_index(drop=True)
            else:
                chunk = grp
            
            # Build graph
            graph = self.converter.convert(chunk)
            if graph is None or 'alert' not in graph.node_types:
                continue
            
            graph = graph.to(self.device)
            
            # Apply augmentation for contrastive learning
            aug_graph = apply_edge_dropout(graph, drop_rate=0.10)
            
            # Guard against empty edge_index_dict after aggressive dropout
            # Check both edge_types existence and that at least one edge type has edges
            if (not aug_graph.edge_types or 
                not any(hasattr(aug_graph[edge_type], 'edge_index') and 
                       aug_graph[edge_type].edge_index.size(1) > 0 
                       for edge_type in aug_graph.edge_types)):
                logger.warning(f"All edges dropped in {dataset_name}, using original graph")
                aug_graph = graph
            
            _, embeddings = self.hgnn(aug_graph)
            
            if 'alert' not in embeddings:
                continue
            
            emb_alert = embeddings['alert']
            
            # Ensure chunk and embeddings match in length
            min_len = min(len(chunk), emb_alert.size(0))
            chunk = chunk.iloc[:min_len]
            emb_alert = emb_alert[:min_len]
            
            # Sanitise embeddings to prevent NaN propagation
            if torch.isnan(emb_alert).any():
                logger.warning(f"NaN values detected in embeddings for {dataset_name}, replacing with zeros")
                emb_alert = torch.nan_to_num(emb_alert, nan=0.0, posinf=1e6, neginf=-1e6)
                emb_alert = torch.clamp(emb_alert, min=-1e4, max=1e4)
            
            campaign_losses = {}
            
            # 1. Temporal contrastive loss (always on)
            if 'timestamp' in chunk.columns:
                # Handle both datetime objects and string timestamps
                timestamps_raw = chunk['timestamp'].values
                if isinstance(timestamps_raw[0], str):
                    timestamps = pd.to_datetime(timestamps_raw).astype(np.int64) // 10**9
                else:
                    timestamps = timestamps_raw.astype(np.int64) // 10**9
                
                # Pass campaign labels for cross-campaign guard when available
                campaign_labels = chunk['campaign_id'].values if 'campaign_id' in chunk.columns else None
                temp_loss = self.compute_temporal_contrastive_loss(emb_alert, timestamps, campaign_labels)
                campaign_losses['temporal'] = temp_loss
            
            # 2. Supervised contrastive loss (weighted)
            if 'campaign_id' in chunk.columns:
                labels = chunk['campaign_id'].values
                indices = chunk.index.tolist()
                weights = self.get_sample_weights(dataset_name, indices)
                
                # Filter out zero-weight samples
                valid_mask = weights > 0
                if valid_mask.sum() >= 2:
                    valid_emb = emb_alert[valid_mask]
                    valid_labels_np = labels[valid_mask.cpu().numpy()]
                    
                    # Remap labels to contiguous range [0..K-1] to prevent CUDA out-of-bounds
                    unique_labels, valid_labels = torch.unique(
                        torch.tensor(valid_labels_np, device=self.device), 
                        return_inverse=True
                    )
                    valid_weights = weights[valid_mask]
                    
                    valid_emb_norm = F.normalize(valid_emb, dim=-1)
                    sup_loss = self.supcon_loss(valid_emb_norm, valid_labels)
                    
                    # Apply per-sample weights (simple weighting by average weight)
                    avg_weight = valid_weights.mean().item()
                    weighted_sup_loss = sup_loss * avg_weight * self.lambda_sup
                    campaign_losses['supervised'] = weighted_sup_loss
            
            # 3. Pseudo-label contrastive loss (if available)
            pseudo_loss = self.compute_pseudo_contrastive_loss(chunk, emb_alert)
            campaign_losses['pseudo'] = pseudo_loss
            
            # Store campaign losses for later accumulation
            batch_campaign_losses.append(campaign_losses)
        
        # Aggregate losses across all campaigns in this batch
        batch_losses = {}
        for campaign_losses in batch_campaign_losses:
            for loss_type, loss_val in campaign_losses.items():
                if loss_type not in batch_losses:
                    batch_losses[loss_type] = []
                if isinstance(loss_val, torch.Tensor):
                    batch_losses[loss_type].append(loss_val)
        
        # Sum losses of the same type across campaigns
        for loss_type in batch_losses:
            if batch_losses[loss_type]:
                batch_losses[loss_type] = torch.stack(batch_losses[loss_type]).sum()
            else:
                batch_losses[loss_type] = torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = sum(v for v in batch_losses.values() if isinstance(v, torch.Tensor))
        
        # Single backward pass for the entire batch
        self.optimizer.zero_grad()
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.hgnn.parameters(), 1.0)
            self.optimizer.step()
        
        # Accumulate per-dataset losses
        for loss_type, loss_val in batch_losses.items():
            val = loss_val.item() if isinstance(loss_val, torch.Tensor) else loss_val
            epoch_losses[loss_type] += val
            dataset_losses[dataset_name][loss_type] += val
        
        return sum(v.item() if isinstance(v, torch.Tensor) else v for v in batch_losses.values())
    
    def train(self, epochs: int, lr: float = 1e-4):
        """
        Main training loop.
        """
        self.optimizer = torch.optim.AdamW(self.hgnn.parameters(), lr=lr, weight_decay=1e-4)
        
        logger.info(f"Starting hybrid training for {epochs} epochs")
        logger.info(f"Datasets: {list(self.datasets.keys())}")
        logger.info(f"Lambda sup: {self.lambda_sup}, Lambda pseudo: {self.lambda_pseudo}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Refresh pseudo-labels periodically (every 5 epochs)
            if self.lambda_pseudo > 0 and epoch % 5 == 0:
                logger.info(f"Refreshing pseudo-labels at epoch {epoch}")
                # We refresh during train_epoch as it iterates through datasets
            
            # Train one epoch
            epoch_losses, dataset_losses = self.train_epoch(epoch)
            
            # Log progress
            loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()])
            logger.info(f"Epoch {epoch+1}/{epochs} ({time.time() - start_time:.1f}s) - {loss_str}")
            
            # Log per-dataset breakdown every 5 epochs
            if (epoch + 1) % 5 == 0:
                logger.info("=== Per-dataset loss breakdown ===")
                for dataset_name, losses in dataset_losses.items():
                    if losses:  # Only log if dataset had losses
                        dataset_loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                        logger.info(f"  {dataset_name}: {dataset_loss_str}")
        
        logger.info("Training complete")


def load_label_metadata(metadata_path: str) -> pd.DataFrame:
    """Load label metadata CSV."""
    return pd.read_csv(metadata_path, quotechar='"', skipinitialspace=True)


def load_datasets(dataset_configs: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Load datasets for hybrid training.
    
    Args:
        dataset_configs: Dict mapping dataset_name -> file_path
        
    Returns:
        Dict mapping dataset_name -> DataFrame
    """
    datasets = {}
    
    for dataset_name, file_path in dataset_configs.items():
        if not Path(file_path).exists():
            logger.warning(f"Dataset not found: {file_path}")
            continue
        
        # Load based on file extension
        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Ensure timestamp column exists
        if 'timestamp' not in df.columns:
            logger.warning(f"No timestamp column in {dataset_name}, generating synthetic timestamps")
            base_time = pd.Timestamp('2024-01-01')
            df['timestamp'] = [base_time + pd.Timedelta(minutes=i*5) for i in range(len(df))]
        
        datasets[dataset_name] = df
        logger.info(f"Loaded {dataset_name}: {len(df)} alerts")
    
    return datasets


def main():
    import torch.serialization
    torch.serialization.add_safe_globals([torch.nn.parameter.UninitializedParameter])
    
    parser = argparse.ArgumentParser(description="Hybrid Adaptive Training v10")
    parser.add_argument('--base_checkpoint', type=str, 
                       default='hgnn_checkpoints/network_v9_v3/network_it_best.pt',
                       help='Base checkpoint to fine-tune')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--output_dir', type=str, default='hgnn_checkpoints/hybrid_v10',
                       help='Output directory for checkpoint')
    parser.add_argument('--lambda_sup', type=float, default=1.0, help='Supervised loss weight')
    parser.add_argument('--lambda_pseudo', type=float, default=0.5, help='Pseudo-label loss weight')
    parser.add_argument('--temporal_window_hours', type=float, default=1.0,
                       help='Temporal window for positive pairs (hours)')
    parser.add_argument('--pseudo_confidence_threshold', type=float, default=0.8,
                       help='Confidence threshold for pseudo-labels')
    
    args = parser.parse_args()
    
    # GPU guard and optimisation
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    else:
        logger.info(f"Using device: {args.device}")
        if args.device == 'cuda':
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            torch.backends.cudnn.benchmark = True
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    dataset_configs = {
        'UNSW-NB15': 'datasets/unsw_nb15/mitre_format.csv',
        'NSL-KDD': 'datasets/nsl_kdd/mitre_format.csv',
        'CICIDS2017': 'datasets/cicids2017/mitre_format.parquet',
        'TON_IoT': 'datasets/TON_IoT/mitre_format.parquet',
        'OpTC': 'datasets/DARPA_OpTC/processed_optc_full.csv',
        'SQTK_SIEM': 'datasets/SQTK_SIEM/mitre_core_format.csv',
    }
    
    datasets = load_datasets(dataset_configs)
    
    # Load label metadata
    metadata_path = 'datasets/label_metadata.csv'
    if not Path(metadata_path).exists():
        logger.error(f"Label metadata not found: {metadata_path}")
        return
    
    label_metadata = load_label_metadata(metadata_path)
    
    # Initialize trainer
    trainer = HybridAdaptiveTrainer(
        datasets=datasets,
        label_metadata=label_metadata,
        device=args.device,
        temporal_window_hours=args.temporal_window_hours,
        pseudo_confidence_threshold=args.pseudo_confidence_threshold,
        lambda_sup=args.lambda_sup,
        lambda_pseudo=args.lambda_pseudo,
    )
    
    # Setup model
    trainer.setup_model(args.base_checkpoint)
    
    # Train
    trainer.train(epochs=args.epochs, lr=args.lr)
    
    # Save checkpoint
    checkpoint_path = output_dir / 'best.pt'
    torch.save({
        'model_state_dict': trainer.hgnn.state_dict(),
        'hidden_dim': trainer.hidden_dim,
        'temperature': trainer.temperature,
        'sup_temperature': trainer.sup_temperature,
        'lambda_sup': args.lambda_sup,
        'lambda_pseudo': args.lambda_pseudo,
        'temporal_window_hours': args.temporal_window_hours,
        'pseudo_confidence_threshold': args.pseudo_confidence_threshold,
    }, checkpoint_path)
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")


if __name__ == '__main__':
    main()
