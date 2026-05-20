#!/usr/bin/env python3
"""
SupCon Fine-tuning for UNSW-NB15 (and any labeled dataset).

Loads an existing HGNN backbone and fine-tunes with Supervised Contrastive Loss
(Khosla et al. 2020). No classification head is needed at inference — GAEC
works on the fine-tuned embeddings directly.

Saves:
  <output_dir>/best.pt          — best fine-tuned checkpoint (by SupCon loss)
  <output_dir>/test_indices.npy — held-out test indices for proper evaluation

This addresses the UNSW-NB15 overfitting problem:
  - network_v9_v3 GAEC gives only 2 clusters / ARI=0.169 because the backbone
    was trained without any UNSW-specific signal (sparse graph → topo loss = 0).
  - SupCon teaches the encoder to separate campaigns in feature space using
    campaign_id labels — without a classification head, so GAEC still works.
  - Expected ARI after fine-tuning: 0.35–0.55 on held-out test split.
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

from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from utils.seed_control import set_seed
from hgnn.hgnn_correlation import MITREHeteroGNN, AlertToGraphConverter
from training.train_on_datasets import (
    apply_edge_dropout,
    apply_feature_augmentation,
    apply_combined_augmentation,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss — Khosla et al. 2020.
    https://arxiv.org/abs/2004.11362

    For each anchor alert, positives = other alerts with the same campaign_id.
    Negatives = all alerts with different campaign_id.

    Features must be L2-normalised before passing in.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,   # [N, D] L2-normalised
        labels: torch.Tensor,     # [N] integer campaign labels
        class_weights: Optional[torch.Tensor] = None,  # [C] per-class weights
    ) -> torch.Tensor:
        N = features.size(0)
        if N < 4:
            return torch.tensor(0.0, device=features.device)

        # Similarity matrix [N, N]
        sim = torch.mm(features, features.T) / self.temperature

        # Positive mask: same label, exclude diagonal
        labels = labels.view(-1, 1)
        mask_pos = (labels == labels.T).float()
        mask_pos.fill_diagonal_(0.0)

        if mask_pos.sum() == 0:
            # No positive pairs — single-class batch, skip
            return torch.tensor(0.0, device=features.device)

        # For numerical stability: subtract row-wise max before exp
        sim_max, _ = sim.detach().max(dim=1, keepdim=True)
        sim = sim - sim_max

        exp_sim = torch.exp(sim)
        # Exclude self from denominator
        eye = torch.eye(N, device=features.device)
        log_prob = sim - torch.log((exp_sim * (1 - eye)).sum(dim=1, keepdim=True) + 1e-8)

        # Mean over positive pairs per anchor
        n_pos = mask_pos.sum(dim=1)
        loss_per_anchor = -(mask_pos * log_prob).sum(dim=1) / n_pos.clamp(min=1.0)
        # Only average over anchors that have at least one positive
        has_pos = (n_pos > 0).float()
        
        # Apply class-balanced weighting if provided
        if class_weights is not None:
            # Get per-anchor weights based on their class
            anchor_weights = class_weights[labels.squeeze()]  # [N]
            weighted_has_pos = has_pos * anchor_weights
            weighted_loss = loss_per_anchor * anchor_weights
            loss = weighted_loss.sum() / weighted_has_pos.sum().clamp(min=1e-8)
        else:
            loss = (loss_per_anchor * has_pos).sum() / has_pos.sum().clamp(min=1.0)
        
        return loss


class SupConProjector(nn.Module):
    """
    Projection head for SupCon loss — 2-layer MLP (128→128→64) with BatchNorm.
    Follows SimCLR/SupCon paper: backbone → projection head → loss.
    BatchNorm stabilizes training and improves representation quality.
    At inference, we use backbone output directly (no projection).
    """

    def __init__(self, in_dim: int = 128, proj_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class CrossCampaignGraphBuilder:
    """
    Builds HeteroData graphs containing alerts from multiple campaigns.

    Unlike per-campaign graph building, edges (shares_ip, temporal_near)
    now span across campaigns — matching inference conditions where the
    engine processes mixed-campaign alert batches.

    This closes the train/infer graph distribution gap that caps ARI at ~0.52.
    """

    def __init__(
        self,
        converter: AlertToGraphConverter,
        alerts_per_campaign: int = 100,
    ):
        self.converter = converter
        self.alerts_per_campaign = alerts_per_campaign

    def build(
        self,
        campaign_groups: list,  # List of (label_int, DataFrame)
    ) -> tuple:  # Returns (HeteroData, Tensor[N] campaign labels)
        """
        Sample `alerts_per_campaign` from each campaign, merge into one
        DataFrame, build a single graph. Returns (graph, labels).
        """
        chunks = []
        label_list = []

        for label_int, grp in campaign_groups:
            n = min(len(grp), self.alerts_per_campaign)
            chunk = grp.sample(n=n).reset_index(drop=True)
            chunks.append(chunk)
            label_list.extend([label_int] * len(chunk))

        merged = pd.concat(chunks, ignore_index=True)
        graph = self.converter.convert(merged)

        # Handle label/node count alignment (converter may drop alerts)
        n_nodes = graph['alert'].x.size(0) if 'alert' in graph.node_types else 0
        if n_nodes < len(label_list):
            labels = torch.tensor(label_list[:n_nodes], dtype=torch.long)
        elif n_nodes > len(label_list):
            # Pad with -1 (will be filtered out in SupCon loss)
            labels = torch.tensor(
                label_list + [-1] * (n_nodes - len(label_list)),
                dtype=torch.long
            )
        else:
            labels = torch.tensor(label_list, dtype=torch.long)

        return graph, labels


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def evaluate_knn_accuracy(
    hgnn: MITREHeteroGNN,
    df: pd.DataFrame,
    label_col: str,
    label_enc: dict,
    converter: AlertToGraphConverter,
    device: str,
    k: int = 5,
    sample_n: int = 2000,
) -> float:
    """
    Compute k-NN accuracy on embeddings as proxy for representation quality.

    Samples `sample_n` alerts from df, runs HGNN (eval mode), collects embeddings,
    computes k-NN accuracy using cosine distance.

    Returns k-NN accuracy in [0, 1].
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    hgnn.eval()
    sample = df.sample(n=min(sample_n, len(df)), random_state=42)
    labels_raw = sample[label_col].map(label_enc).values

    all_z, all_y = [], []
    for _, grp in sample.groupby(label_col):
        graph = converter.convert(grp)
        if graph is None or 'alert' not in graph.node_types or len(graph.edge_types) == 0:
            continue
        try:
            with torch.no_grad():
                _, emb = hgnn(graph.to(device))
            z = F.normalize(emb['alert'], dim=-1).cpu().numpy()
            y = [label_enc[lbl] for lbl in grp[label_col]]
            all_z.append(z)
            all_y.extend(y)
        except Exception as e:
            continue

    if len(all_z) < 2:
        return 0.0

    Z = np.vstack(all_z)
    Y = np.array(all_y)

    # 5-fold cross-val k-NN
    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    scores = cross_val_score(knn, Z, Y, cv=min(5, len(set(Y))), scoring='accuracy')
    return float(scores.mean())


# ---------------------------------------------------------------------------

def finetune_supcon(
    dataset_path: str = 'datasets/unsw_nb15/mitre_format.csv',
    label_col: str = 'campaign_id',
    base_checkpoint: str = 'hgnn_checkpoints/network_v9_v3/network_it_best.pt',
    output_dir: str = 'hgnn_checkpoints/unsw_supcon_v5',
    epochs: int = 120,
    lr: float = 1e-4,
    temperature: float = 0.07,
    test_size: float = 0.25,
    batch_size: int = 4,
    aug_drop_rate: float = 0.10,
    device: str = 'cuda',
    max_chunk: int = 400,
    use_projection: bool = False,
    cross_campaign: bool = False,
    alerts_per_campaign: int = 1000,
    steps_per_epoch: int = 16,
    early_stop_patience: int = 20,
    use_dual_view: bool = True,
    feat_mask_rate: float = 0.20,
    feat_noise_std: float = 0.05,
    class_balanced: bool = True,
    class_weight_cap: float = 4.0,
) -> None:
    """
    Fine-tune HGNN backbone with SupCon loss on a labeled dataset.

    Args
    ----
    dataset_path        Path to mitre_format.csv (or .parquet).
    label_col           Column containing integer/string campaign labels.
    base_checkpoint     Pretrained backbone to load (network_v9_v3 for UNSW-NB15).
    output_dir          Directory for best.pt and test_indices.npy.
    epochs              Number of fine-tuning epochs.
    lr                  AdamW learning rate.
    temperature         SupCon temperature (0.07 is optimal for UNSW-NB15, 0.05 too aggressive).
    test_size           Fraction of alerts held out for evaluation.
    batch_size          Campaigns processed per gradient step.
    aug_drop_rate       Edge dropout rate for augmentation.
    device              'cuda' or 'cpu'.
    max_chunk           Max alerts per campaign chunk (prevents OOM).
    use_projection      If True, use 2-layer MLP projection head before loss (SupCon paper).
    cross_campaign      If True, build graphs with mixed campaigns (matches inference).
    alerts_per_campaign Alerts sampled per campaign in cross-campaign mode.
    steps_per_epoch     Gradient steps per epoch in cross-campaign mode.
    early_stop_patience Stop training if no improvement for N epochs (default 20).
    use_dual_view       If True, use dual-view augmentation (two independent views per batch).
    feat_mask_rate      Fraction of alert features to zero-mask per augmentation (default 0.20).
    feat_noise_std      Std dev of Gaussian noise added to continuous features (default 0.05).
    class_balanced       Enable class-balanced SupCon loss (default True).
    class_weight_cap     Cap class weights to prevent instability (default 4.0).
    """
    set_seed(42)
    logger.info(f"SupCon fine-tuning: {dataset_path}")
    logger.info(f"Base checkpoint: {base_checkpoint}")
    logger.info(f"Output dir: {output_dir}")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    p = Path(dataset_path)
    if p.suffix == '.parquet':
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    if df.empty:
        raise ValueError(f"Empty dataset: {dataset_path}")

    # Encode labels to contiguous ints
    unique_labels = sorted(df[label_col].unique())
    label_enc = {v: i for i, v in enumerate(unique_labels)}
    df = df.copy()
    df['_label_int'] = df[label_col].map(label_enc)
    n_classes = len(unique_labels)
    logger.info(f"Dataset: {len(df)} alerts, {n_classes} campaigns: {unique_labels}")

    # ------------------------------------------------------------------
    # Filter out classes with too few samples for stratified split
    # ------------------------------------------------------------------
    class_counts = df['_label_int'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    if len(valid_classes) < len(class_counts):
        dropped_classes = class_counts[class_counts < 2].index
        logger.warning(f"Dropping {len(dropped_classes)} classes with < 2 samples: {dropped_classes.tolist()}")
        df = df[df['_label_int'].isin(valid_classes)].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Stratified train/test split — save test indices
    # ------------------------------------------------------------------
    all_idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=test_size,
        stratify=df['_label_int'].values,
        random_state=42,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.save(output_path / 'test_indices.npy', test_idx)
    logger.info(f"Test indices saved: {output_path / 'test_indices.npy'} ({len(test_idx)} samples)")

    train_df = df.iloc[train_idx].reset_index(drop=True)
    logger.info(f"Training split: {len(train_df)} alerts")

    # ------------------------------------------------------------------
    # Compute class-balanced weights if enabled
    # ------------------------------------------------------------------
    class_weights = None
    if class_balanced:
        # Get training labels for weight computation
        train_labels = np.array([label_enc.get(label, -1) for label in train_df[label_col].fillna("UNKNOWN").astype(str)])
        # Filter out any -1 labels (shouldn't happen but safety)
        valid_mask = train_labels >= 0
        train_labels = train_labels[valid_mask]
        n_classes = len(label_enc)
        
        # Compute inverse-frequency weights
        counts = np.bincount(train_labels, minlength=n_classes).astype(float)
        weights = counts.sum() / (len(counts) * np.clip(counts, 1.0, None))  # mean=1
        weights = np.clip(weights, 1.0/class_weight_cap, class_weight_cap)  # cap extremes
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        
        logger.info(f"Class weights: {dict(zip([str(k) for k in label_enc.keys()], weights.round(3).tolist()))}")
        logger.info(f"Class distribution: {dict(zip([str(k) for k in label_enc.keys()], counts.astype(int).tolist()))}")
    else:
        logger.info("Class-balanced loss disabled")

    # ------------------------------------------------------------------
    # Load backbone checkpoint
    # ------------------------------------------------------------------
    ckpt = torch.load(base_checkpoint, map_location='cpu', weights_only=False)
    hidden_dim = ckpt.get('hidden_dim', 128)

    # Read num_clusters from checkpoint to avoid shape mismatch
    num_clusters = 10  # default
    if 'model_state_dict' in ckpt:
        for key in ckpt['model_state_dict']:
            if 'cluster_classifier.3.weight' in key:
                num_clusters = ckpt['model_state_dict'][key].shape[0]
                break

    # Enable CategoricalAlertEncoder for SupCon to handle feature encoding properly
    vocab_sizes = ckpt.get('vocab_sizes', None)
    if vocab_sizes:
        logger.info(f"Using CategoricalAlertEncoder with vocab_sizes: {vocab_sizes}")
    else:
        logger.info("No vocab_sizes in checkpoint, using Linear encoder")

    hgnn = MITREHeteroGNN(
        alert_feature_dim=15,  # Must match network_v9_v3's alert_encoder weight shape [128, 15]
        hidden_dim=hidden_dim,
        num_layers=1,
        num_clusters=num_clusters,
        vocab_sizes=vocab_sizes,  # Enable CategoricalAlertEncoder if available
    ).to(device)

    missing, unexpected = hgnn.load_state_dict(ckpt['model_state_dict'], strict=False)
    if missing:
        logger.warning(f"Missing keys when loading backbone: {missing[:5]}")
    if unexpected:
        logger.warning(f"Unexpected keys when loading backbone: {unexpected[:5]}")
    logger.info(f"Backbone loaded from {base_checkpoint} (hidden_dim={hidden_dim}, num_clusters={num_clusters})")

    # ------------------------------------------------------------------
    # Projection head (for SupCon loss only)
    # ------------------------------------------------------------------
    projector = SupConProjector(in_dim=hidden_dim, proj_dim=64).to(device) if use_projection else None
    if use_projection:
        logger.info("Using projection head (128→128→64) before SupCon loss")

    # ------------------------------------------------------------------
    # Optimizer and Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(hgnn.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = SupConLoss(temperature=temperature)

    converter = AlertToGraphConverter(temporal_window_hours=1.0, build_bridge_edges=False)
    builder = CrossCampaignGraphBuilder(converter, alerts_per_campaign=alerts_per_campaign)

    if use_projection:
        optimizer = torch.optim.AdamW(
            list(hgnn.parameters()) + list(projector.parameters()),
            lr=lr, weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.AdamW(hgnn.parameters(), lr=lr, weight_decay=1e-4)

    # Warmup + cosine decay scheduler
    warmup_epochs = 10
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    # Group training data by campaign for positive-pair batching
    campaign_groups = [
        (label, grp.reset_index(drop=True))
        for label, grp in train_df.groupby('_label_int')
        if len(grp) >= 4
    ]
    logger.info(f"Valid campaign groups (≥4 alerts): {len(campaign_groups)}")

    if cross_campaign:
        logger.info(f"Cross-campaign mode: {alerts_per_campaign} alerts/campaign, {steps_per_epoch} steps/epoch")
        builder = CrossCampaignGraphBuilder(
            converter=converter,
            alerts_per_campaign=alerts_per_campaign,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_loss = float('inf')
    patience_counter = 0
    train_start = time.time()
    epoch_iter = tqdm(range(epochs), desc="SupCon", unit="epoch") if TQDM_AVAILABLE else range(epochs)

    for epoch in epoch_iter:
        hgnn.train()
        epoch_loss = 0.0
        n_steps = 0

        if cross_campaign:
            # Cross-campaign mode: build mixed graphs with cross-campaign edges
            supcon = SupConLoss(temperature=temperature)
            for step_idx in range(steps_per_epoch):
                # Shuffle campaign order each step for diversity
                random.shuffle(campaign_groups)

                graph, labels = builder.build(campaign_groups)
                if graph is None or 'alert' not in graph.node_types:
                    continue
                # Pad alert features 6→15 to match backbone alert_encoder.weight [128, 15]
                if graph['alert'].x.shape[1] < 15:
                    graph['alert'].x = torch.nn.functional.pad(
                        graph['alert'].x, (0, 15 - graph['alert'].x.shape[1])
                    )
                graph = graph.to(device)
                labels = labels.to(device)

                aug = apply_edge_dropout(graph, drop_rate=aug_drop_rate)
                _, emb = hgnn(aug)

                if 'alert' not in emb:
                    continue

                z = F.normalize(emb['alert'], dim=-1)

                # Trim labels to match actual node count (convert() may drop alerts)
                if z.size(0) != labels.size(0):
                    labels = labels[:z.size(0)]

                # Filter out padding labels (-1)
                valid_mask = (labels >= 0)
                if valid_mask.sum() < 4:
                    continue
                z = z[valid_mask]
                labels = labels[valid_mask]

                if use_projection:
                    z = F.normalize(projector(z), dim=-1)

                loss = supcon(z, labels, class_weights)
                if loss.item() == 0.0:
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hgnn.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_steps += 1
        else:
            # Original per-campaign mode (unchanged)
            # Shuffle campaign order each epoch
            random.shuffle(campaign_groups)

            for batch_start in range(0, len(campaign_groups), batch_size):
                batch_campaigns = campaign_groups[batch_start:batch_start + batch_size]
                if not batch_campaigns:
                    continue

                z_parts, label_parts = [], []

                for label_int, grp in batch_campaigns:
                    # Sample ONE random chunk of max_chunk alerts (not all chunks).
                    # This keeps z_all size = batch_size * max_chunk = ~1600 embeddings,
                    # preventing the N*N similarity matrix from OOM on a 8GB GPU.
                    if len(grp) > max_chunk:
                        chunk = grp.sample(n=max_chunk, random_state=None).reset_index(drop=True)
                    else:
                        chunk = grp

                    if len(chunk) < 4:
                        continue

                    graph = converter.convert(chunk)
                    if graph is None or 'alert' not in graph.node_types:
                        continue

                    # Pad alert features 6→15 to match backbone alert_encoder.weight [128, 15]
                    if graph['alert'].x.shape[1] < 15:
                        graph['alert'].x = torch.nn.functional.pad(
                            graph['alert'].x, (0, 15 - graph['alert'].x.shape[1])
                        )

                    graph_base = graph.to(device)

                    # --- DUAL-VIEW SUPCON (use_dual_view=True, default) ---
                    if use_dual_view:
                        # Two INDEPENDENT augmentations of the same campaign
                        aug1 = apply_combined_augmentation(
                            graph_base, edge_drop_rate=aug_drop_rate,
                            feat_mask_rate=feat_mask_rate, feat_noise_std=feat_noise_std
                        )
                        aug2 = apply_combined_augmentation(
                            graph_base, edge_drop_rate=aug_drop_rate,
                            feat_mask_rate=feat_mask_rate, feat_noise_std=feat_noise_std
                        )

                        try:
                            _, emb1 = hgnn(aug1)
                            _, emb2 = hgnn(aug2)
                        except Exception as e:
                            logger.debug(f"Forward pass failed for campaign {label_int}: {e}")
                            continue

                        if 'alert' not in emb1 or 'alert' not in emb2:
                            continue

                        n1 = emb1['alert'].size(0)
                        n2 = emb2['alert'].size(0)
                        min_n = min(n1, n2)

                        z1 = F.normalize(emb1['alert'][:min_n], dim=-1)
                        z2 = F.normalize(emb2['alert'][:min_n], dim=-1)

                        if use_projection:
                            z1 = F.normalize(projector(z1), dim=-1)
                            z2 = F.normalize(projector(z2), dim=-1)

                        # Concatenate dual views: [z1; z2], labels repeated twice
                        z_cat = torch.cat([z1, z2], dim=0)
                        lbl = torch.full((min_n,), label_int, dtype=torch.long, device=device)
                        lbl_cat = torch.cat([lbl, lbl], dim=0)

                        z_parts.append(z_cat)
                        label_parts.append(lbl_cat)
                    else:
                        # Original single-view mode (for backward compatibility)
                        aug = apply_edge_dropout(graph_base, drop_rate=aug_drop_rate)

                        try:
                            _, emb = hgnn(aug)
                        except Exception as e:
                            logger.debug(f"Forward pass failed for campaign {label_int}: {e}")
                            continue

                        if 'alert' not in emb:
                            continue

                        z = F.normalize(emb['alert'], dim=-1)   # [n_alerts, D]
                        n_alerts = z.size(0)
                        z_parts.append(z)
                        label_parts.append(
                            torch.full((n_alerts,), label_int, dtype=torch.long, device=device)
                        )

                if not z_parts:
                    continue

                z_all = torch.cat(z_parts, dim=0)           # [N_total, D]  (≤ batch_size*max_chunk)
                labels_all = torch.cat(label_parts, dim=0)  # [N_total]

                loss = supcon(z_all, labels_all, class_weights)
                if loss.item() == 0.0:
                    continue

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping for both backbone and projector
                if use_projection:
                    torch.nn.utils.clip_grad_norm_(list(hgnn.parameters()) + list(projector.parameters()), 1.0)
                else:
                    torch.nn.utils.clip_grad_norm_(hgnn.parameters(), 1.0)

                optimizer.step()

                epoch_loss += loss.item()
                n_steps += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - train_start
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)

        if avg_loss < best_loss and n_steps > 0:
            best_loss = avg_loss
            patience_counter = 0  # Reset patience on improvement
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': hgnn.state_dict(),
                'loss': best_loss,
                'hidden_dim': hidden_dim,
                'n_classes': n_classes,
                'label_enc': label_enc,
                'cross_campaign': cross_campaign,
                'use_dual_view': use_dual_view,
            }
            if use_projection:
                checkpoint['projector_state_dict'] = projector.state_dict()
            torch.save(checkpoint, output_path / 'best.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"Early stopping: no improvement for {early_stop_patience} epochs")
                break

        status = (
            f"Epoch {epoch+1}/{epochs}: SupCon={avg_loss:.4f} | best={best_loss:.4f} "
            f"| steps={n_steps} | ETA={eta/60:.1f}min"
        )
        if TQDM_AVAILABLE:
            epoch_iter.set_postfix_str(
                f"loss={avg_loss:.4f} best={best_loss:.4f} ETA={eta/60:.1f}min"
            )
        logger.info(status)

    total_time = time.time() - train_start
    logger.info(f"Fine-tuning complete in {total_time/60:.1f}min. Best SupCon loss: {best_loss:.4f}")
    logger.info(f"Checkpoint : {output_path / 'best.pt'}")
    logger.info(f"Test indices: {output_path / 'test_indices.npy'} ({len(test_idx)} samples)")

    # Evaluate k-NN accuracy on training set embeddings
    logger.info("Evaluating k-NN accuracy on training set embeddings...")
    knn_acc = evaluate_knn_accuracy(hgnn, train_df, label_col, label_enc, converter, device)
    logger.info(f"k-NN (k=5) accuracy on training embeddings: {knn_acc:.3f}")

    logger.info("")
    logger.info("Next step — update DATASET_CONFIG in run_gate_tuning.py:")
    logger.info(f'  "checkpoint_override": "{output_dir}/best.pt",')
    logger.info(f'  "test_indices_path":   "{output_dir}/test_indices.npy",')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='SupCon fine-tuning for UNSW-NB15 (or any labeled MITRE-format dataset)'
    )
    parser.add_argument(
        '--dataset_path',
        default='datasets/unsw_nb15/mitre_format.csv',
        help='Path to mitre_format.csv or .parquet',
    )
    parser.add_argument('--label_col', default='campaign_id')
    parser.add_argument(
        '--base_checkpoint',
        default='hgnn_checkpoints/network_v9_v3/network_it_best.pt',
        help='Backbone checkpoint to fine-tune from',
    )
    parser.add_argument(
        '--output_dir',
        default='hgnn_checkpoints/unsw_supcon_v5',
        help='Directory for best.pt and test_indices.npy',
    )
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='SupCon temperature (0.07 is optimal for UNSW-NB15, 0.05 too aggressive)')
    parser.add_argument('--test_size', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Campaigns per gradient step')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--use_projection', action='store_true', default=False,
                        help='Use projection head before SupCon loss')
    parser.add_argument('--cross_campaign', action='store_true', default=False,
                        help='Build cross-campaign graphs (mix campaigns before graph construction)')
    parser.add_argument('--alerts_per_campaign', type=int, default=1000,
                        help='Alerts sampled per campaign in cross-campaign mode')
    parser.add_argument('--steps_per_epoch', type=int, default=16,
                        help='Gradient steps per epoch in cross-campaign mode')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='Stop training if no improvement for N epochs')
    parser.add_argument('--use_dual_view', action='store_true', default=True,
                        help='Use dual-view augmentation (TWO independent views per batch step)')
    parser.add_argument('--feat_mask_rate', type=float, default=0.20,
                        help='Fraction of alert features to zero-mask per augmentation')
    parser.add_argument('--feat_noise_std', type=float, default=0.05,
                        help='Std dev of Gaussian noise added to continuous features')
    parser.add_argument('--class_balanced', action='store_true', default=True,
                        help='Enable class-balanced SupCon loss')
    parser.add_argument('--no_class_balanced', dest='class_balanced', action='store_false',
                        help='Disable class-balanced SupCon loss')
    parser.add_argument('--class_weight_cap', type=float, default=4.0,
                        help='Cap class weights to prevent instability')

    args = parser.parse_args()
    finetune_supcon(**vars(args))
