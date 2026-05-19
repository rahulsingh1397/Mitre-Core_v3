"""
Supervised Contrastive Loss implementation for MITRE-CORE training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss - Khosla et al. 2020.
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
            # No positive pairs - single-class batch, skip
            return torch.tensor(0.0, device=features.device)

        # For numerical stability: subtract row-wise max before exp
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Exp similarity
        exp_sim = torch.exp(sim)

        # Log-sum-exp denominator (all except diagonal)
        mask_all = torch.ones_like(mask_pos)
        mask_all.fill_diagonal_(0.0)
        log_prob = sim - torch.log(exp_sim * mask_all).sum(dim=1, keepdim=True)

        # Keep only positive pairs
        log_prob = log_prob * mask_pos

        # Mean over positive pairs
        num_pos = mask_pos.sum()
        if num_pos == 0:
            return torch.tensor(0.0, device=features.device)

        loss = -log_prob.sum() / num_pos
        return loss
