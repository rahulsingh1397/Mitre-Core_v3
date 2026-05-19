#!/usr/bin/env python3
"""
Fast multi-dataset training script for network_v9_v2 (optimized version)
Skips semantic similarity edge computation for faster training.
"""

import logging
import random
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
import pandas as pd
import sys
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
sys.path.append('.')
from utils.seed_control import set_seed
from hgnn.hgnn_correlation import MITREHeteroGNN
from training.train_on_datasets import PublicDatasetGraphConverter, apply_edge_dropout
from hgnn.cross_domain_contrastive import CrossDomainContrastiveLoss, CrossGraphNTXentLoss
from hgnn.categorical_encoder import compute_vocab_sizes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def topological_ntxent_loss(
    embeddings: torch.Tensor,          # [N, D] alert embeddings for this graph
    graph: HeteroData,                  # full graph with all edge types
    temperature: float = 0.1,          # lower T = sharper boundaries
    n_negatives: int = 256,            # random negatives to sample
) -> torch.Tensor:
    """
    Fully vectorized GPU implementation of topological NT-Xent loss.
    For each alert, its positive is any alert it shares an edge with.
    Negatives are randomly sampled alerts.
    """
    n = embeddings.size(0)
    if n < 4:
        return torch.tensor(0.0, device=embeddings.device)

    device = embeddings.device
    
    # Step 1: Build adjacency mask from edge indices (no .item() calls)
    adj = torch.zeros(n, n, dtype=torch.bool, device=device)
    for et in graph.edge_types:
        src_type, rel, dst_type = et
        if src_type == 'alert' and dst_type == 'alert' and et in graph.edge_index_dict:
            ei = graph.edge_index_dict[et]
            # Sample edges if too many
            max_edges = 1000
            if ei.size(1) > max_edges:
                indices = torch.randperm(ei.size(1), device=device)[:max_edges]
                ei = ei[:, indices]
            
            # Direct tensor indexing - no loops, no .item()
            adj[ei[0], ei[1]] = True
            adj[ei[1], ei[0]] = True  # symmetric
    
    adj.fill_diagonal_(False)  # Remove self-connections
    
    # Step 2: Sample anchor-positive pairs via tensor ops
    pos_src, pos_dst = adj.nonzero(as_tuple=True)
    n_pairs = len(pos_src)
    
    if n_pairs == 0:
        return torch.tensor(0.0, device=device)
    
    # Sample up to max_pairs
    max_pairs = 512
    if n_pairs > max_pairs:
        idx = torch.randperm(n_pairs, device=device)[:max_pairs]
        pos_src = pos_src[idx]
        pos_dst = pos_dst[idx]
        n_pairs = max_pairs
    
    # Step 3: Batch negative sampling
    # For each anchor, sample n_neg random nodes (may include some positives - acceptable approximation)
    n_neg = min(n_negatives, n - 1)
    neg_indices = torch.randint(0, n, (n_pairs, n_neg), device=device)
    
    # Step 4: Single batched InfoNCE
    z = torch.nn.functional.normalize(embeddings, dim=-1)
    
    anchor_emb = z[pos_src]           # [P, D]
    pos_emb = z[pos_dst]              # [P, D]
    neg_emb = z[neg_indices]          # [P, n_neg, D]
    
    # Compute similarities
    pos_sim = (anchor_emb * pos_emb).sum(-1, keepdim=True) / temperature  # [P, 1]
    neg_sim = torch.bmm(neg_emb, anchor_emb.unsqueeze(-1)).squeeze(-1) / temperature  # [P, n_neg]
    
    # Single batched cross_entropy for all pairs
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # [P, 1+n_neg]
    labels = torch.zeros(n_pairs, dtype=torch.long, device=device)
    
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss

# Network-IT datasets for joint training
NETWORK_IT_DATASETS = ['unsw_nb15', 'nsl_kdd', 'ton_iot', 'optc']

def train_graph_mae_v9_multidata_fast(epochs=150, output_dir='./hgnn_checkpoints/network_v9_v3', 
                                     batch_size=8, temperature_aug=0.5, temperature_topo=0.2, 
                                     hidden_dim=128, device='cuda', use_cross_graph=False,
                                     topo_weight=1.0, simclr_weight=0.5, seed=42):
    """Fast joint training on multiple Network-IT datasets without semantic similarity edges.

    Hybrid loss: ``loss = topo_weight * topological_NT_Xent + simclr_weight * SimCLR``.
    Defaults (1.0, 0.5) reproduce the canonical network_v9_v3 67/33 ratio after
    normalisation. Set ``topo_weight=1.0, simclr_weight=0.0`` for pure topological,
    or ``topo_weight=0.0, simclr_weight=1.0`` for pure SimCLR — used in the §7.2
    hybrid-loss ablation.
    """
    
    set_seed(seed)  # Reproducibility (seed configurable for multi-seed sweeps)
    logger.info(f"Starting fast network_v9_v2 training on {NETWORK_IT_DATASETS}")
    
    # Initialize converter (skip trainer since we're doing manual training)
    converter = PublicDatasetGraphConverter(build_bridge_edges=True)
    
    all_graphs = []
    
    for ds_name in NETWORK_IT_DATASETS:
        logger.info(f"Loading {ds_name}...")
        # Load dataset manually
        # Special handling for OpTC dataset
        if ds_name == 'optc':
            dataset_path = Path("datasets/DARPA_OpTC/processed_optc_full.csv")
        else:
            dataset_path = Path(f"datasets/{ds_name}/mitre_format.csv")
        if not dataset_path.exists():
            logger.warning(f"Could not find {dataset_path}, skipping.")
            continue
            
        df = pd.read_csv(dataset_path)
        if df.empty:
            logger.warning(f"Empty dataset {ds_name}, skipping.")
            continue
            
        # Use all data for training (no train/test split needed for self-supervised)
        train_df = df.copy()
        
        # Handle OpTC dataset which uses CampaignId instead of campaign_id
        campaign_col = 'CampaignId' if ds_name == 'optc' else 'campaign_id'
        if campaign_col not in train_df.columns:
            logger.warning(f"No {campaign_col} column in {ds_name}, creating single campaign")
            train_df[campaign_col] = 0  # Single campaign for all alerts
        
        # Group alerts by campaign_id with chunking to avoid OOM
        graph_count = 0
        for campaign_id, campaign_df in train_df.groupby(campaign_col):
            if len(campaign_df) < 10:
                continue
                
            # Use chunks of up to 500 alerts to avoid OOM
            for chunk_start in range(0, len(campaign_df), 500):
                chunk_df = campaign_df.iloc[chunk_start:chunk_start + 500]
                graph = converter.convert(chunk_df)
                if graph is not None and 'alert' in graph.node_types:
                    all_graphs.append(graph)
                    graph_count += 1
                    if graph_count % 50 == 0:
                        logger.info(f"Processed {graph_count} graphs from {ds_name}...")
    
    random.shuffle(all_graphs)
    
    # Pre-move all graphs to GPU to eliminate per-step CPU→GPU transfer
    logger.info(f"Moving {len(all_graphs)} graphs to {device}...")
    all_graphs = [g.to(device) for g in all_graphs]
    logger.info("Graphs loaded to GPU.")
    
    logger.info(f"Joint training pool: {len(all_graphs)} graphs from {NETWORK_IT_DATASETS}")

    # Compute vocabulary sizes for categorical features
    dataset_paths = {ds: f"datasets/{ds}/mitre_format.csv" for ds in NETWORK_IT_DATASETS}
    vocab_sizes = compute_vocab_sizes(NETWORK_IT_DATASETS, dataset_paths)
    logger.info(f"Computed vocab sizes: {vocab_sizes}")

    # Initialize model WITH learned embeddings
    hgnn = MITREHeteroGNN(
        alert_feature_dim=6,  # B1: 6-dim base features (matches PublicDatasetGraphConverter output)
        hidden_dim=hidden_dim,
        num_layers=1,
        num_clusters=10,
        vocab_sizes=vocab_sizes,  # Use learned embeddings
    ).to(device)
    
    # Optimizer with AdamW and weight decay
    optimizer = torch.optim.AdamW(
        hgnn.parameters(),
        lr=3e-4,
        weight_decay=1e-4       # L2 regularization prevents collapse
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Contrastive loss functions
    contrastive_fn = CrossDomainContrastiveLoss(temperature=temperature_aug)
    cross_graph_fn = CrossGraphNTXentLoss(temperature=temperature_topo) if use_cross_graph else None
    
    # Training loop
    best_loss = float('inf')
    output_path = Path(output_dir) / "network_it_best.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting fast network_v9_v2 Training ({epochs} epochs) on device={device}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train_start = time.time()
    epoch_iter = tqdm(range(epochs), desc="Training", unit="epoch") if TQDM_AVAILABLE else range(epochs)

    for epoch in epoch_iter:
        hgnn.train()

        total_topo_loss = 0.0
        total_aug_loss = 0.0
        total_loss = 0.0
        n_steps = 0
        epoch_start = time.time()

        # Process batches of graphs
        for i in range(0, len(all_graphs), batch_size):
            batch_graphs = all_graphs[i:i+batch_size]

            optimizer.zero_grad()

            # Dual-augmentation approach (proven to work)
            z1_parts, z2_parts, topo_losses = [], [], []
            batch_connected_pairs = []

            for graph in batch_graphs:
                # Create two augmentations with edge dropout
                aug1 = apply_edge_dropout(graph, drop_rate=0.15)
                aug2 = apply_edge_dropout(graph, drop_rate=0.15)

                # HGNN forward on both augmentations
                _, emb1 = hgnn(aug1)
                _, emb2 = hgnn(aug2)

                z1_parts.append(emb1['alert'])
                z2_parts.append(emb2['alert'])

                # Use optimized topological loss
                if not use_cross_graph:
                    # Optimized topological NT-Xent loss
                    topo_loss = topological_ntxent_loss(emb1['alert'], aug1, temperature=temperature_topo)
                    topo_losses.append(topo_loss)
                else:
                    # Gather connected pairs for cross-graph loss
                    connected_pairs = set()
                    n = emb1['alert'].size(0)
                    for et in aug1.edge_types:
                        src_type, rel, dst_type = et
                        if src_type == 'alert' and dst_type == 'alert' and et in aug1.edge_index_dict:
                            ei = aug1.edge_index_dict[et]
                            for k in range(ei.size(1)):
                                i_, j_ = ei[0, k].item(), ei[1, k].item()
                                if i_ != j_ and i_ < n and j_ < n:
                                    connected_pairs.add((min(i_, j_), max(i_, j_)))
                    batch_connected_pairs.append(connected_pairs)

            # Dual-augmentation NT-Xent (campaign-level, original)
            z1 = torch.cat(z1_parts, dim=0)
            z2 = torch.cat(z2_parts, dim=0)
            aug_ntxent = contrastive_fn(z1, z2, "train", "train")

            # Combined loss (topological primary, augmentation secondary)
            if use_cross_graph and cross_graph_fn is not None:
                avg_topo = cross_graph_fn(z1_parts, batch_connected_pairs)
            else:
                avg_topo = torch.stack(topo_losses).mean() if topo_losses else torch.tensor(0.0, device=device)
            
            loss = topo_weight * avg_topo + simclr_weight * aug_ntxent

            loss.backward()
            torch.nn.utils.clip_grad_norm_(hgnn.parameters(), 1.0)
            optimizer.step()

            total_topo_loss += avg_topo.item()
            total_aug_loss += aug_ntxent.item()
            total_loss += loss.item()
            n_steps += 1

        scheduler.step()

        avg_topo = total_topo_loss / max(n_steps, 1)
        avg_aug = total_aug_loss / max(n_steps, 1)
        avg_total = total_loss / max(n_steps, 1)
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save({
                'epoch': epoch,
                'model_state_dict': hgnn.state_dict(),
                'loss': best_loss,
                'hidden_dim': hidden_dim,
                'vocab_sizes': vocab_sizes,  # Save for inference compatibility
            }, output_path)

        status = f"Epoch {epoch+1}/{epochs}: Topo={avg_topo:.4f}, Aug={avg_aug:.4f}, Total={avg_total:.4f} | {epoch_time:.1f}s/ep | ETA={eta/60:.1f}min"

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(output_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': hgnn.state_dict(),
                'loss': avg_total,
                'best_loss': best_loss,
                'hidden_dim': hidden_dim,
                'vocab_sizes': vocab_sizes,  # Save for inference compatibility
            }, checkpoint_path)
            status += " [ckpt saved]"

        if TQDM_AVAILABLE:
            epoch_iter.set_postfix_str(f"Topo={avg_topo:.4f} Aug={avg_aug:.4f} Best={best_loss:.4f} ETA={eta/60:.1f}min")
        logger.info(status)
            
    logger.info(f"Training complete. Best loss: {best_loss:.4f}")
    logger.info(f"Model saved to {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--output_dir", type=str, default="./hgnn_checkpoints/network_v9_v5_contextual")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cross_graph", action="store_true", help="Enable cross-graph NT-Xent loss")
    parser.add_argument("--topo_weight", type=float, default=1.0,
                        help="Topological NT-Xent loss weight (default 1.0; canonical 67/33 ratio)")
    parser.add_argument("--simclr_weight", type=float, default=0.5,
                        help="SimCLR augmentation NT-Xent loss weight (default 0.5; canonical 67/33 ratio)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (use {42,43,44} for multi-seed sweeps)")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_graph_mae_v9_multidata_fast(
        epochs=args.epochs,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=device,
        use_cross_graph=args.cross_graph,
        topo_weight=args.topo_weight,
        simclr_weight=args.simclr_weight,
        seed=args.seed,
    )
