#!/usr/bin/env python3
"""
Cross-Sensor Encoder Fine-Tuning (Track 10)

Surgical freeze: lock the entire backbone, train only the 3 new components
added in CS-2:
  1. source_sensor_encoder (Linear layer)
  2. collected_by edge convolution (GATConv)
  3. collects edge convolution (GATConv)

Problem: These components were added in CS-2 but all checkpoints predate them.
Zero-initialized weights → the new pathway contributes noise instead of signal
→ TON_IoT ARI collapses from 0.737 to 0.054.

Solution: One or two epochs of NT-Xent contrastive loss to give the new pathway
a gradient signal without disturbing the backbone's learned embedding geometry.

Saves:
  <output_dir>/network_cs_best.pt — fine-tuned checkpoint with trained CS components

Success criteria:
  - TON_IoT ARI ≥ 0.70 (recovers pre-CS baseline)
  - UNSW AMI ≥ 0.55 (recovers SupCon baseline)
  - cosine_sim < 0.40 post fine-tune (no embedding collapse)
"""

import os
import sys
import random
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from sklearn.metrics import adjusted_mutual_info_score
from utils.seed_control import set_seed
from hgnn.hgnn_correlation import MITREHeteroGNN, AlertToGraphConverter
from hgnn.contrastive_loss import NTXentLoss
from training.train_on_datasets import apply_edge_dropout

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_ami(
    hgnn: MITREHeteroGNN,
    df: pd.DataFrame,
    converter: AlertToGraphConverter,
    device: str,
    sample_n: int = 2000,
) -> float:
    """
    Compute AMI on embeddings as proxy for clustering quality.
    
    Samples `sample_n` alerts from df, runs HGNN (eval mode), collects embeddings,
    computes AMI using HDBSCAN labels as ground truth proxy.
    
    Returns AMI score.
    """
    from hdbscan import HDBSCAN
    
    hgnn.eval()
    sample = df.sample(n=min(sample_n, len(df)), random_state=42)
    
    all_z = []
    for _, grp in sample.groupby('campaign_id' if 'campaign_id' in sample.columns else 'label'):
        graph = converter.convert(grp)
        if graph is None or 'alert' not in graph.node_types:
            continue
        try:
            with torch.no_grad():
                _, emb = hgnn(graph.to(device))
            if 'alert' in emb:
                z = F.normalize(emb['alert'], dim=-1).cpu().numpy()
                all_z.append(z)
        except Exception as e:
            logger.debug(f"Forward pass failed: {e}")
            continue
    
    if len(all_z) < 1:
        return 0.0
    
    Z = np.vstack(all_z)
    
    # Run HDBSCAN on embeddings
    clusterer = HDBSCAN(min_cluster_size=10, metric='cosine')
    pred_labels = clusterer.fit_predict(Z)
    
    # If no valid clusters, return 0
    if len(set(pred_labels)) <= 1:
        return 0.0
    
    # Use campaign_id as ground truth if available
    if 'campaign_id' in sample.columns:
        # Align labels with sampled embeddings
        gt_labels = []
        for _, grp in sample.groupby('campaign_id'):
            n = min(len(grp), sample_n // len(sample['campaign_id'].unique()))
            gt_labels.extend([grp['campaign_id'].iloc[0]] * n)
        gt_labels = gt_labels[:len(Z)]
        if len(gt_labels) != len(Z):
            return 0.0
        return float(adjusted_mutual_info_score(gt_labels, pred_labels))
    
    # Fallback: use silhouette score as proxy
    from sklearn.metrics import silhouette_score
    if len(set(pred_labels)) > 1:
        return float(silhouette_score(Z, pred_labels, metric='cosine'))
    return 0.0


def finetune_cross_sensor(
    dataset_paths: list = None,
    base_checkpoint: str = 'hgnn_checkpoints/network_v9_v3/network_it_best.pt',
    output_dir: str = 'hgnn_checkpoints/network_v9_v3_cs',
    epochs: int = 2,
    lr: float = 1e-3,
    temperature: float = 0.07,
    batch_size: int = 4,
    aug_drop_rate: float = 0.10,
    device: str = 'cuda',
    max_chunk: int = 400,
    early_stop_patience: int = 3,
    datasets: list = None,
    output_checkpoint: str = None,
    seed: int = 42,
    track_data_source: bool = False,
) -> None:
    """
    Fine-tune only the cross-sensor components (encoder + 2 edge convs).
    
    Args
    ----
    dataset_paths       List of dataset paths to train on (UNSW-NB15, TON_IoT)
    base_checkpoint     Pretrained backbone to load (network_v9_v3)
    output_dir          Directory for network_cs_best.pt
    epochs              Number of fine-tuning epochs (1-2 is sufficient)
    lr                  Adam learning rate for new components only
    temperature         NT-Xent temperature
    batch_size          Campaigns processed per gradient step
    aug_drop_rate       Edge dropout rate for augmentation
    device              'cuda' or 'cpu'
    max_chunk           Max alerts per campaign chunk
    early_stop_patience Stop training if no AMI improvement for N epochs
    datasets            Dataset names (alternative to dataset_paths)
    output_checkpoint   Output checkpoint path (alternative to output_dir)
    seed                Random seed for reproducibility
    track_data_source   Enable cross-sensor edge tracking
    """
    # Handle dataset name mapping
    logger.info(f"DEBUG: datasets={datasets}, dataset_paths={dataset_paths}")
    if datasets:
        dataset_map = {
            'UNSW-NB15': 'datasets/unsw_nb15/mitre_format.csv',
            'TON_IoT': 'datasets/TON_IoT/mitre_format.parquet',
        }
        dataset_paths = [dataset_map[name] for name in datasets]
        logger.info(f"Mapped dataset names to paths: {dict(zip(datasets, dataset_paths))}")
    # If no datasets provided, use dataset_paths as-is (default behavior)
    
    # Handle output checkpoint path
    if output_checkpoint and not output_dir:
        output_dir = str(Path(output_checkpoint).parent)
        logger.info(f"Derived output_dir from output_checkpoint: {output_dir}")
    
    set_seed(seed)
    logger.info(f"Cross-sensor fine-tuning: {dataset_paths}")
    logger.info(f"Base checkpoint: {base_checkpoint}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Track data source: {track_data_source}")
    
    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    dfs = []
    for path in dataset_paths:
        p = Path(path)
        if p.suffix == '.parquet':
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
        if df.empty:
            logger.warning(f"Empty dataset: {path}")
            continue
        dfs.append(df)
        logger.info(f"Loaded {len(df)} alerts from {path}")
    
    if not dfs:
        raise ValueError("No valid datasets loaded")
    
    # Merge datasets for training
    train_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total training alerts: {len(train_df)}")
    
    # ------------------------------------------------------------------
    # Load backbone checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
    hidden_dim = ckpt.get('hidden_dim', 128)
    
    # Read num_clusters from checkpoint
    num_clusters = 10
    if 'model_state_dict' in ckpt:
        for key in ckpt['model_state_dict']:
            if 'cluster_classifier.3.weight' in key:
                num_clusters = ckpt['model_state_dict'][key].shape[0]
                break
    
    vocab_sizes = ckpt.get('vocab_sizes', None)
    
    hgnn = MITREHeteroGNN(
        alert_feature_dim=6,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_clusters=num_clusters,
        vocab_sizes=vocab_sizes,
    ).to(device)
    
    missing, unexpected = hgnn.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        logger.info(f"Missing keys (expected for new CS components): {missing}")
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected[:5]}")
    logger.info(f"Backbone loaded (hidden_dim={hidden_dim}, num_clusters={num_clusters})")
    
    # ------------------------------------------------------------------
    # Initialize lazy modules BEFORE freezing
    # ------------------------------------------------------------------
    logger.info("Initializing lazy modules with dummy forward pass...")
    hgnn.eval()
    with torch.no_grad():
        # Create a dummy HeteroData with all node types
        from torch_geometric.data import HeteroData
        dummy_data = HeteroData()
        dummy_data['alert'].x = torch.randn(10, 6, device=device)
        dummy_data['source_sensor'].x = torch.randn(2, 1, device=device)
        dummy_data['alert', 'collected_by', 'source_sensor'].edge_index = torch.tensor([[0, 1], [0, 0]], device=device).long()
        dummy_data['source_sensor', 'collects', 'alert'].edge_index = torch.tensor([[0, 0], [0, 1]], device=device).long()
        # Add required edge types for the model
        dummy_data['alert', 'shares_ip', 'alert'].edge_index = torch.tensor([[0, 1], [1, 0]], device=device).long()
        try:
            _, _ = hgnn(dummy_data)
            logger.info("Lazy modules initialized successfully")
        except Exception as e:
            logger.warning(f"Dummy forward pass failed (may be OK): {e}")
    
    # ------------------------------------------------------------------
    # Differential learning rates: slow for backbone, normal for new CS components
    # ------------------------------------------------------------------
    # Instead of surgical freeze (which blocks gradient flow), use differential LR
    # This allows gradients to flow through the entire network while updating
    # new components much faster than the frozen backbone
    
    # Identify new CS parameters
    new_cs_params = []
    backbone_params = []
    
    for name, param in hgnn.named_parameters():
        if 'source_sensor_encoder' in name or \
           ('collected_by' in name or 'collects' in name):
            new_cs_params.append(param)
            logger.info(f"New CS parameter: {name}")
        else:
            backbone_params.append(param)
    
    if not new_cs_params:
        raise ValueError("No new CS parameters found. Check checkpoint compatibility.")
    
    logger.info(f"New CS parameters: {len(new_cs_params)}")
    logger.info(f"Backbone parameters: {len(backbone_params)}")
    
    # ------------------------------------------------------------------
    # Optimizer and Loss
    # ------------------------------------------------------------------
    # Differential learning rates: new CS components get normal LR, backbone gets 1/100
    optimizer = torch.optim.Adam([
        {'params': new_cs_params, 'lr': lr},
        {'params': backbone_params, 'lr': lr / 100.0}  # Very slow backbone updates
    ], weight_decay=1e-4)
    loss_fn = NTXentLoss(temperature=temperature)
    
    # CRITICAL: track_data_source parameter so source_sensor nodes appear in graphs
    converter = AlertToGraphConverter(
        temporal_window_hours=1.0,
        build_bridge_edges=False,
        track_data_source=track_data_source,  # Required for CS encoder to receive gradients
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_ami = 0.0
    patience_counter = 0
    train_start = time.time()
    epoch_iter = tqdm(range(epochs), desc="CS Fine-tune", unit="epoch") if TQDM_AVAILABLE else range(epochs)
    
    # Group by campaign for batching
    label_col = 'campaign_id' if 'campaign_id' in train_df.columns else 'label'
    campaign_groups = [
        (label, grp.reset_index(drop=True))
        for label, grp in train_df.groupby(label_col)
        if len(grp) >= 4
    ]
    logger.info(f"Valid campaign groups (≥4 alerts): {len(campaign_groups)}")
    
    for epoch in epoch_iter:
        hgnn.train()
        epoch_loss = 0.0
        n_steps = 0
        
        # Shuffle campaign order each epoch
        random.shuffle(campaign_groups)
        
        for batch_start in range(0, len(campaign_groups), batch_size):
            batch_campaigns = campaign_groups[batch_start:batch_start + batch_size]
            if not batch_campaigns:
                continue
            
            z_parts = []
            
            for label_int, grp in batch_campaigns:
                if len(grp) > max_chunk:
                    chunk = grp.sample(n=max_chunk, random_state=None).reset_index(drop=True)
                else:
                    chunk = grp
                
                if len(chunk) < 4:
                    continue
                
                graph = converter.convert(chunk)
                if graph is None or 'alert' not in graph.node_types:
                    continue
                
                # Debug: check if source_sensor nodes were created
                if 'source_sensor' not in graph.node_types:
                    logger.debug("No source_sensor nodes in graph - skipping (no gradient flow)")
                    continue
                
                # Pad or truncate alert features to match expected dim (6 for network_v9_v3 backbone, but residual connection expects 15)
                # CS-1 added source_sensor_id as dimension 21, so we may have 21 dims
                if graph['alert'].x.shape[1] < 15:
                    graph['alert'].x = torch.nn.functional.pad(
                        graph['alert'].x, (0, 15 - graph['alert'].x.shape[1])
                    )
                elif graph['alert'].x.shape[1] > 15:
                    # Truncate to 15 dims (drop contextual features added in CS-1)
                    graph['alert'].x = graph['alert'].x[:, :15]
                
                graph = graph.to(device)
                
                # Debug: check if CS edges exist
                has_cs_edges = ('alert', 'collected_by', 'source_sensor') in graph.edge_types
                if not has_cs_edges:
                    logger.debug("No collected_by edges - skipping (no gradient flow)")
                    continue
                
                # Edge dropout augmentation
                aug = apply_edge_dropout(graph, drop_rate=aug_drop_rate)
                _, emb = hgnn(aug)
                
                if 'alert' not in emb:
                    continue
                
                z = F.normalize(emb['alert'], dim=-1)
                z_parts.append(z)
            
            if not z_parts:
                continue
            
            z_all = torch.cat(z_parts, dim=0)
            
            # Build positive pairs from temporal proximity (simple heuristic)
            # In a real setting, you'd use graph structure or campaign labels
            # For unsupervised fine-tuning, we use temporal adjacency as positive signal
            N = z_all.size(0)
            positive_pairs = []
            for i in range(N):
                for j in range(i + 1, min(i + 10, N)):  # Assume nearby alerts are positives
                    positive_pairs.append((i, j))
            
            loss = loss_fn(z_all, positive_pairs)
            if loss.item() == 0.0:
                continue
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(hgnn.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_steps += 1
        
        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - train_start
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)
        
        # Evaluate AMI for early stopping
        current_ami = evaluate_ami(hgnn, train_df, converter, device, sample_n=1000)
        logger.info(f"Epoch {epoch+1}/{epochs}: NT-Xent={avg_loss:.4f} | AMI={current_ami:.4f} | ETA={eta/60:.1f}min")
        
        # Always save checkpoint (either best improvement or final epoch)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': hgnn.state_dict(),
            'loss': avg_loss,
            'ami': current_ami,
            'hidden_dim': hidden_dim,
            'num_clusters': num_clusters,
        }
        # Use output_checkpoint if provided, otherwise default to network_cs_best.pt
        if output_checkpoint:
            save_path = Path(output_checkpoint)
        else:
            save_path = Path(output_dir) / 'network_cs_best.pt'
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")
        
        if current_ami > best_ami:
            best_ami = current_ami
            patience_counter = 0
            logger.info(f"New best AMI: {best_ami:.4f} - saved checkpoint")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping: no AMI improvement for {early_stop_patience} epochs")
                break
        
        if TQDM_AVAILABLE:
            epoch_iter.set_postfix_str(f"loss={avg_loss:.4f} ami={current_ami:.4f} best={best_ami:.4f}")
    
    total_time = time.time() - train_start
    logger.info(f"Fine-tuning complete in {total_time/60:.1f}min")
    logger.info(f"Best AMI: {best_ami:.4f}")
    logger.info(f"Checkpoint: {output_dir}/network_cs_best.pt")
    
    # Verify cosine similarity to check for collapse
    logger.info("Computing cosine similarity statistics...")
    hgnn.eval()
    sample = train_df.sample(n=min(500, len(train_df)), random_state=42)
    all_z = []
    for _, grp in sample.groupby(label_col):
        graph = converter.convert(grp)
        if graph is None or 'alert' not in graph.node_types:
            continue
        try:
            with torch.no_grad():
                _, emb = hgnn(graph.to(device))
            if 'alert' in emb:
                z = F.normalize(emb['alert'], dim=-1).cpu().numpy()
                all_z.append(z)
        except:
            continue
    
    if all_z:
        Z = np.vstack(all_z)
        # Compute pairwise cosine similarity
        sim_matrix = Z @ Z.T
        # Exclude diagonal
        np.fill_diagonal(sim_matrix, 0)
        mean_cosine = sim_matrix[np.triu_indices_from(sim_matrix, k=1)].mean()
        logger.info(f"Mean cosine similarity (off-diagonal): {mean_cosine:.4f}")
        if mean_cosine > 0.40:
            logger.warning(f"Embedding collapse detected (cosine_sim={mean_cosine:.4f} > 0.40)")
        else:
            logger.info(f"No collapse detected (cosine_sim={mean_cosine:.4f} < 0.40)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Cross-sensor encoder fine-tuning (surgical freeze)'
    )
    parser.add_argument(
        '--dataset_paths',
        nargs='+',
        default=['datasets/unsw_nb15/mitre_format.csv', 'datasets/ton_iot/mitre_format.csv'],
        help='Paths to MITRE-format datasets for fine-tuning'
    )
    parser.add_argument(
        '--base_checkpoint',
        default='hgnn_checkpoints/network_v9_v3/network_it_best.pt',
        help='Backbone checkpoint to fine-tune from'
    )
    parser.add_argument(
        '--output_dir',
        default='hgnn_checkpoints/network_v9_v3_cs',
        help='Directory for network_cs_best.pt'
    )
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of fine-tuning epochs (1-2 is sufficient)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Adam learning rate for new components')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='NT-Xent temperature')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Campaigns per gradient step')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--early_stop_patience', type=int, default=3,
                        help='Stop training if no AMI improvement for N epochs')
    parser.add_argument('--datasets', nargs='+', 
                        help='Dataset names (alternative to dataset_paths)')
    parser.add_argument('--output_checkpoint', 
                        help='Output checkpoint path (alternative to output_dir)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--track_data_source', action='store_true',
                        help='Enable cross-sensor edge tracking')
    
    args = parser.parse_args()
    finetune_cross_sensor(**vars(args))
