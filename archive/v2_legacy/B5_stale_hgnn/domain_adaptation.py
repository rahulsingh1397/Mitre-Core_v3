"""
Domain Adaptation
=================
Implements Gradient Reversal Layer (GRL) and Domain Discriminator for
adversarial domain alignment (DANN style).
"""

import torch
import torch.nn as nn
from torch.autograd import Function

class GradientReversalLayer(Function):
    """
    Reverses the gradient during the backward pass.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainDiscriminator(nn.Module):
    """
    Binary or multi-class domain classifier on top of HGNN backbone embeddings.
    """
    def __init__(self, hidden_dim: int, num_domains: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_domains)
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Forward pass with gradient reversal.
        """
        x_rev = GradientReversalLayer.apply(x, alpha)
        return self.net(x_rev)
