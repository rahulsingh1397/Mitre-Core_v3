"""
Heterogeneous Variational Graph Autoencoder (VGAE) for MITRE-CORE
=================================================================
Pre-trains the MITREHeteroGNN backbone using link prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, Tuple

class HeteroVGAE(nn.Module):
    """
    Variational Graph Autoencoder for heterogeneous alert graphs.
    Uses MITREHeteroGNN as the encoder.
    """
    def __init__(self, encoder: nn.Module, hidden_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        
        # Linear layers for mu and logvar
        self.mu = nn.Linear(hidden_dim, hidden_dim)
        self.logvar = nn.Linear(hidden_dim, hidden_dim)
        
    def encode(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode graph into latent representations.
        Returns: (mu, logvar) for alert nodes.
        """
        # Forward pass through backbone
        _, x_dict = self.encoder(data)
        
        alert_embeds = x_dict['alert']
        mu = self.mu(alert_embeds)
        logvar = self.logvar(alert_embeds)
        
        return mu, logvar
        
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
            
    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Dot-product decoder for link prediction.
        Args:
            z: Latent alert representations [N, D]
            edge_index: Edges to predict [2, E]
        Returns:
            Logits for each edge [E]
        """
        src = z[edge_index[0]]
        dst = z[edge_index[1]]
        return (src * dst).sum(dim=-1)
        
    def forward(self, data: HeteroData, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            data: Input graph
            edge_index: Edges to reconstruct (typically positive edges)
        """
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, edge_index)
        return logits, mu, logvar
        
    def kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Kullback-Leibler divergence loss"""
        return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
