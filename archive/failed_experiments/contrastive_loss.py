"""
Contrastive loss implementations for MITRE-CORE training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import List, Tuple


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss for contrastive learning.
    
    This is the standard InfoNCE loss used in SimCLR and related works.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.similarity_f = nn.CosineSimilarity(dim=2)
    
    def forward(self, embeddings: torch.Tensor, positive_pairs: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Compute NT-Xent loss with given positive pairs using vectorised operations.
        
        Args:
            embeddings: Tensor of shape (N, D) with normalized embeddings
            positive_pairs: List of (i, j) positive pair indices
            
        Returns:
            NT-Xent loss scalar
        """
        if len(positive_pairs) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        N = embeddings.size(0)
        
        # Build positive mask matrix
        pos_mask = torch.zeros(N, N, dtype=torch.bool, device=embeddings.device)
        for i, j in positive_pairs:
            pos_mask[i, j] = True
            pos_mask[j, i] = True  # Symmetric
        
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings, embeddings.t()) / self.temperature  # (N, N)
        
        # Mask out diagonal (self-similarity)
        diag_mask = torch.eye(N, dtype=torch.bool, device=embeddings.device)
        sim_matrix.masked_fill_(diag_mask, float('-inf'))
        
        # Apply log_softmax row-wise
        log_prob = F.log_softmax(sim_matrix, dim=1)
        
        # Compute loss: -log(exp(sim_ij) / sum_k exp(sim_ik)) for positive pairs
        # Since we already applied log_softmax, this is simply -log_prob for positive pairs
        pos_log_prob = log_prob[pos_mask]
        
        if pos_log_prob.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        loss = -pos_log_prob.mean()
        return loss


class TemporalNTXentLoss(nn.Module):
    """
    NT-Xent loss with temporal positive pairs.
    """
    
    def __init__(self, temperature: float = 0.07, temporal_window_hours: float = 2.0):
        super().__init__()
        self.temperature = temperature
        self.temporal_window_hours = temporal_window_hours
        self.base_loss = NTXentLoss(temperature)
    
    def forward(self, embeddings: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal NT-Xent loss.
        
        Args:
            embeddings: Tensor of shape (N, D) with alert embeddings
            timestamps: Tensor of shape (N,) with Unix timestamps
            
        Returns:
            Temporal NT-Xent loss scalar
        """
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=-1)
        
        # Build temporal positive pairs
        positive_pairs = self._build_temporal_pairs(timestamps)
        
        return self.base_loss(embeddings_norm, positive_pairs)
    
    def _build_temporal_pairs(self, timestamps: torch.Tensor) -> List[Tuple[int, int]]:
        """Build positive pairs based on temporal proximity."""
        window_seconds = self.temporal_window_hours * 3600
        positive_pairs = []
        
        for i in range(len(timestamps)):
            for j in range(i + 1, len(timestamps)):
                if abs(timestamps[i] - timestamps[j]) <= window_seconds:
                    positive_pairs.append((i, j))
        
        return positive_pairs


class TopologicalNTXentLoss(nn.Module):
    """
    Topological NT-Xent loss using graph adjacency for positive pairs.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings: torch.Tensor, graph: HeteroData) -> torch.Tensor:
        """
        Compute topological NT-Xent loss.
        
        Args:
            embeddings: Tensor of shape (N, D) with alert embeddings
            graph: HeteroData graph with alert-to-alert edges
            
        Returns:
            Topological NT-Xent loss scalar
        """
        N = embeddings.size(0)
        if N < 4:
            return torch.tensor(0.0, device=embeddings.device)
        
        device = embeddings.device
        
        # Build adjacency mask from alert-to-alert edges
        adj = torch.zeros(N, N, dtype=torch.bool, device=device)
        
        # Add edges from alert-to-alert connections
        if ('alert', 'connected_to', 'alert') in graph.edge_types:
            edge_index = graph['alert', 'connected_to', 'alert'].edge_index
            adj[edge_index[0], edge_index[1]] = True
            adj[edge_index[1], edge_index[0]] = True  # Symmetric
        
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=-1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t()) / self.temperature
        
        # Create positive pairs from adjacency
        positive_pairs = []
        for i in range(N):
            for j in range(i + 1, N):
                if adj[i, j]:
                    positive_pairs.append((i, j))
        
        if len(positive_pairs) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute NT-Xent loss with these pairs
        loss = 0.0
        for i, j in positive_pairs:
            # Anchor i
            sim_i = sim_matrix[i]
            pos_sim = sim_i[j]
            
            # Negatives (all except i)
            neg_mask = torch.ones(N, dtype=torch.bool, device=device)
            neg_mask[i] = False
            neg_sims = sim_i[neg_mask]
            
            # Log-sum-exp
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            loss += -F.log_softmax(logits, dim=0)[0]
        
        return loss / len(positive_pairs)
