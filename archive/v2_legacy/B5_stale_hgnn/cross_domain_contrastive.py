"""
Cross-Domain Contrastive Loss
=============================
NT-Xent loss with MITRE-technique-aware positive/negative mining.
Positives: (alert_i, alert_j) where same MITRE technique T-ID,
           even if from different domains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Tuple, List
import random

logger = logging.getLogger("mitre-core.contrastive")

class CrossGraphNTXentLoss(nn.Module):
    """
    NT-Xent loss with cross-graph true negatives.
    
    Instead of only using intra-graph negatives (which can lead to campaign-level
    collapse), this loss uses alerts from OTHER graphs in the same batch as guaranteed
    true negatives, forcing the model to learn campaign-discriminative features.
    """
    def __init__(self, temperature: float = 0.1, n_negatives: int = 256):
        super().__init__()
        self.temperature = temperature
        self.n_negatives = n_negatives
        
    def forward(
        self, 
        batch_embeddings: List[torch.Tensor],  # List of [N_g, D] alert embeddings per graph
        batch_connected_pairs: List[set]       # List of sets of (i, j) positive pairs per graph
    ) -> torch.Tensor:
        """
        Args:
            batch_embeddings: List of B tensors, each [N_g, D] containing alert embeddings for graph g
            batch_connected_pairs: List of B sets, each containing tuples of connected alert indices for graph g
            
        Returns:
            Scalar NT-Xent loss
        """
        B = len(batch_embeddings)
        if B < 2:
            # Cannot do cross-graph negatives with only 1 graph
            return torch.tensor(0.0, device=batch_embeddings[0].device)
            
        # Normalize all embeddings
        norm_embeddings = [F.normalize(emb, dim=-1) for emb in batch_embeddings]
        device = norm_embeddings[0].device
        
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        
        for g_idx in range(B):
            z_g = norm_embeddings[g_idx]
            n_g = z_g.size(0)
            pairs = list(batch_connected_pairs[g_idx])
            
            if not pairs:
                continue
                
            # Sample pairs if too many
            if len(pairs) > 128:
                pairs = random.sample(pairs, 128)
                
            # Gather all negatives from OTHER graphs
            cross_graph_negs = []
            for other_g_idx in range(B):
                if other_g_idx == g_idx:
                    continue
                cross_graph_negs.append(norm_embeddings[other_g_idx])
                
            if not cross_graph_negs:
                continue
                
            all_cross_negs = torch.cat(cross_graph_negs, dim=0)  # [Total_Other_N, D]
            n_other = all_cross_negs.size(0)
            
            if n_other == 0:
                continue
                
            for (anchor_idx, pos_idx) in pairs:
                anchor = z_g[anchor_idx]  # [D]
                pos = z_g[pos_idx]        # [D]
                
                # Sample negatives from other graphs
                n_neg = min(self.n_negatives, n_other)
                if n_neg < n_other:
                    neg_indices = torch.randperm(n_other, device=device)[:n_neg]
                    negs = all_cross_negs[neg_indices]  # [n_neg, D]
                else:
                    negs = all_cross_negs
                    
                pos_sim = (anchor * pos).sum() / self.temperature
                neg_sims = torch.mv(negs, anchor) / self.temperature  # [n_neg]
                
                logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])  # [1 + n_neg]
                label = torch.tensor(0, device=device)  # positive is index 0
                
                total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))
                count += 1
                
        return total_loss / max(count, 1)

class CrossDomainContrastiveLoss(nn.Module):
    """
    NT-Xent loss with cross-domain technique-based positive mining.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor, 
                domain_i: str, domain_j: str,
                tech_i: Optional[torch.Tensor] = None, 
                tech_j: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute cross-domain contrastive loss.
        """
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        batch_size = z_i.size(0)
        
        # Representations: [2*N, D]
        z = torch.cat([z_i, z_j], dim=0)
        
        # Similarity matrix: [2*N, 2*N]
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        
        # Create mask for self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -9e15)
        
        # If technique IDs are provided, use them for supervised contrastive
        if tech_i is not None and tech_j is not None:
            techs = torch.cat([tech_i, tech_j], dim=0)
            # Positive mask: same technique
            pos_mask = (techs.unsqueeze(0) == techs.unsqueeze(1)) & ~mask
            
            # For numerical stability
            max_sim = torch.max(sim_matrix, dim=1, keepdim=True)[0]
            exp_sim = torch.exp(sim_matrix - max_sim)
            
            # Sum of exp similarities for positives
            pos_sum = (exp_sim * pos_mask).sum(dim=1)
            # Sum of exp similarities for all (including negatives)
            all_sum = exp_sim.sum(dim=1)
            
            # Loss
            loss = -torch.log(pos_sum / (all_sum + 1e-8) + 1e-8).mean()
        else:
            # Standard SimCLR (positive pairs are corresponding augmented views)
            pos_sim = torch.cat([
                torch.diag(sim_matrix, batch_size),
                torch.diag(sim_matrix, -batch_size)
            ])
            
            # For numerical stability
            max_sim = torch.max(sim_matrix, dim=1)[0]
            
            numerator = torch.exp(pos_sim - max_sim)
            denominator = torch.exp(sim_matrix - max_sim.unsqueeze(1)).sum(dim=1)
            
            loss = -torch.log(numerator / denominator).mean()
            
        return loss
