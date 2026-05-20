"""
GraphMAE Objective for MITRE-CORE
Implements masked autoencoding for alert graphs to prevent embedding collapse.
Reference: Hou et al. "GraphMAE" (KDD 2022) - scaled cosine error variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Tuple


class MaskedAlertDecoder(nn.Module):
    """Decoder for masked alert feature reconstruction."""
    
    def __init__(self, embed_dim: int = 128, feat_dim: int = 6):
        super().__init__()
        self.decoder = nn.Linear(embed_dim, feat_dim)
    
    def forward(self, z: torch.Tensor, original_feats: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Alert embeddings [N, embed_dim]
            original_feats: Original alert features [N, feat_dim]
            mask_indices: Indices of masked nodes [mask_ratio * N]
        
        Returns:
            Scaled cosine reconstruction loss
        """
        pred = self.decoder(z[mask_indices])  # [masked, feat_dim]
        target = original_feats[mask_indices]  # [masked, feat_dim]
        
        # Scaled cosine error (GraphMAE variant)
        pred_norm = F.normalize(pred, dim=-1)
        target_norm = F.normalize(target, dim=-1)
        
        # Cosine similarity -> convert to error (1 - cos)
        cosine_sim = (pred_norm * target_norm).sum(-1)  # [masked]
        loss = (1.0 - cosine_sim).mean()  # scalar
        
        return loss


def mask_alert_features(feat_tensor: torch.Tensor, mask_ratio: float = 0.40) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly mask alert features for GraphMAE pre-training.
    
    Args:
        feat_tensor: Alert features [N, feat_dim]
        mask_ratio: Fraction of nodes to mask (default 0.40)
    
    Returns:
        masked_feats: Features with masked nodes set to zero [N, feat_dim]
        mask_indices: Indices of masked nodes [mask_ratio * N]
    """
    n = feat_tensor.size(0)
    num_masked = int(n * mask_ratio)
    
    # Randomly select nodes to mask
    mask_indices = torch.randperm(n)[:num_masked]
    
    # Create masked features
    masked_feats = feat_tensor.clone()
    masked_feats[mask_indices] = 0.0
    
    return masked_feats, mask_indices


def graph_mae_loss(
    embeddings: torch.Tensor, 
    original_features: torch.Tensor, 
    decoder: MaskedAlertDecoder,
    mask_ratio: float = 0.40
) -> torch.Tensor:
    """
    Compute GraphMAE loss for a batch of alert embeddings.
    
    Args:
        embeddings: HGNN alert embeddings [N, embed_dim]
        original_features: Original alert features before masking [N, feat_dim]
        decoder: MaskedAlertDecoder module
        mask_ratio: Fraction of nodes to mask
    
    Returns:
        GraphMAE reconstruction loss
    """
    masked_feats, mask_indices = mask_alert_features(original_features, mask_ratio)
    return decoder(embeddings, original_features, mask_indices)


def neighborhood_mae_loss(
    graph: HeteroData, 
    alert_emb: torch.Tensor,
    decoder: nn.Module, 
    mask_ratio: float = 0.40
) -> torch.Tensor:
    """
    MAE target = mean of neighbors' raw features (from full graph).
    Forces GNN to encode structural position, not just feature values.
    
    Args:
        graph: Full HeteroData graph with all node types and edges
        alert_emb: Alert embeddings from HGNN [N, embed_dim]
        decoder: MaskedAlertDecoder module
        mask_ratio: Fraction of nodes to mask
    
    Returns:
        Neighborhood reconstruction loss
    """
    n = alert_emb.size(0)
    mask_n = max(1, int(n * mask_ratio))
    mask_idx = torch.randperm(n, device=alert_emb.device)[:mask_n]

    # Aggregate neighbor features for each masked alert (including cross-type neighbors)
    neighbor_targets = []
    alert_feats = graph['alert'].x  # [N, 6]
    for idx in mask_idx:
        idx_val = idx.item()
        neighbors = []
        for et in graph.edge_types:
            src_type, _, dst_type = et
            if dst_type == 'alert' and et in graph.edge_index_dict:
                ei = graph.edge_index_dict[et]
                # Find neighbors pointing to this alert
                mask_edges = (ei[1] == idx_val)
                src_nodes = ei[0][mask_edges]
                if src_nodes.numel() > 0:
                    if src_type == 'alert':
                        # Alert neighbors - use alert features
                        neighbors.append(alert_feats[src_nodes])
                    elif src_type in graph.node_types and hasattr(graph[src_type], 'x'):
                        # Cross-type neighbors (user, host) - use their features
                        # Map to same 6-dim space by taking first 6 dims or padding
                        src_feats = graph[src_type].x[src_nodes]
                        if src_feats.shape[1] >= 6:
                            neighbors.append(src_feats[:, :6])  # Take first 6 dims
                        else:
                            # Pad smaller features to 6 dims
                            padding = torch.zeros(src_nodes.shape[0], 6 - src_feats.shape[1], device=src_feats.device)
                            padded = torch.cat([src_feats, padding], dim=1)
                            neighbors.append(padded)
        
        if neighbors:
            target = torch.cat(neighbors, dim=0).mean(dim=0)
        else:
            target = alert_feats[idx_val]  # fallback: self
        neighbor_targets.append(target)

    targets = torch.stack(neighbor_targets, dim=0)  # [mask_n, 6]
    preds = decoder.decoder(alert_emb[mask_idx])       # [mask_n, 6] - call decoder directly
    pred_n = F.normalize(preds, dim=-1)
    tgt_n  = F.normalize(targets, dim=-1)
    return (1.0 - (pred_n * tgt_n).sum(-1)).mean()
