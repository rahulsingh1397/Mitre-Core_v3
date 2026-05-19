"""
CategoricalAlertEncoder for MITRE-CORE
======================================

Replaces raw integer-coded categorical features with learned embeddings
and scalar temporal features with cyclical sin/cos encoding.

This addresses the structural embedding scatter problem where:
- Protocol codes 0-132 (129 unique in UNSW campaigns 34/48) scatter embeddings
- Hour/day_of_week linear normalization loses cyclical structure

Author: MITRE-CORE Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CategoricalAlertEncoder(nn.Module):
    """
    Encoder for alert features with learned embeddings for categoricals
    and cyclical encoding for temporal features.

    Input: [N, 15] tensor where:
        - dim 0: tactic (integer category code)
        - dim 1: alert_type (binary 0/1)
        - dim 2: hour (raw 0-23, will be cyclically encoded)
        - dim 3: day_of_week (raw 0-6, will be cyclically encoded)
        - dim 4: protocol (integer category code)
        - dim 5: service (integer category code)
        - dims 6-14: contextual features (9-dim, passthrough)

    Output: [N, hidden_dim] encoded alert representations
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_tactics: int = 50,
        n_protocols: int = 200,
        n_services: int = 50,
        tactic_embed_dim: int = 8,
        protocol_embed_dim: int = 16,
        service_embed_dim: int = 8,
        n_contextual: int = 9,
    ):
        """
        Args:
            hidden_dim: Final output dimension (typically 128 for HGNN)
            n_tactics: Vocabulary size for tactic embeddings (padding_idx=0 reserved)
            n_protocols: Vocabulary size for protocol embeddings (padding_idx=0 reserved)
            n_services: Vocabulary size for service embeddings (padding_idx=0 reserved)
            tactic_embed_dim: Dimension of learned tactic embeddings
            protocol_embed_dim: Dimension of learned protocol embeddings
            service_embed_dim: Dimension of learned service embeddings
            n_contextual: Number of contextual feature dimensions (default 9)
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_contextual = n_contextual

        # Learned embedding tables for categorical features
        # padding_idx=0 means code 0 (missing/unknown) maps to zero vector
        self.tactic_embed = nn.Embedding(n_tactics, tactic_embed_dim, padding_idx=0)
        self.protocol_embed = nn.Embedding(n_protocols, protocol_embed_dim, padding_idx=0)
        self.service_embed = nn.Embedding(n_services, service_embed_dim, padding_idx=0)

        # Store embedding dims for forward pass
        self.tactic_embed_dim = tactic_embed_dim
        self.protocol_embed_dim = protocol_embed_dim
        self.service_embed_dim = service_embed_dim

        # Calculate total input dimension to projection
        # tactic_embed + protocol_embed + service_embed + hour_sin/cos + dow_sin/cos + alert_type + contextual
        self.total_input_dim = (
            tactic_embed_dim
            + protocol_embed_dim
            + service_embed_dim
            + 2  # hour sin/cos
            + 2  # day_of_week sin/cos
            + 1  # alert_type
            + n_contextual
        )

        # Projection to hidden_dim with LayerNorm and ReLU
        self.projection = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Initialize embeddings with small random values
        self._init_weights()

    def _init_weights(self):
        """Initialize embedding tables with small random values."""
        with torch.no_grad():
            # Uniform initialization in [-0.1, 0.1] for embeddings
            for embed in [self.tactic_embed, self.protocol_embed, self.service_embed]:
                nn.init.uniform_(embed.weight, -0.1, 0.1)
                # Ensure padding_idx=0 remains zero
                if embed.padding_idx is not None:
                    embed.weight[embed.padding_idx].fill_(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode alert features with learned embeddings.

        Args:
            x: [N, 15] tensor with base features (dims 0-5) and contextual (dims 6-14)

        Returns:
            [N, hidden_dim] encoded representations
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        n = x.size(0)

        # Extract categorical codes (cast to long for embedding lookup)
        # Clamp to valid range to handle out-of-vocab codes gracefully
        tactic_codes = x[:, 0].long().clamp(0, self.tactic_embed.num_embeddings - 1)
        protocol_codes = x[:, 4].long().clamp(0, self.protocol_embed.num_embeddings - 1)
        # Handle 5-dim input (no service feature) by using zeros
        if x.size(1) >= 6:
            service_codes = x[:, 5].long().clamp(0, self.service_embed.num_embeddings - 1)
        else:
            service_codes = torch.zeros(n, dtype=torch.long, device=x.device)

        # Lookup embeddings
        tactic_emb = self.tactic_embed(tactic_codes)  # [N, tactic_embed_dim]
        protocol_emb = self.protocol_embed(protocol_codes)  # [N, protocol_embed_dim]
        service_emb = self.service_embed(service_codes)  # [N, service_embed_dim]

        # Extract and cyclically encode temporal features
        # x[:, 2] is hour (raw 0-23), x[:, 3] is day_of_week (raw 0-6)
        hour = x[:, 2]
        dow = x[:, 3]

        # Cyclical encoding: sin(2π * value / period), cos(2π * value / period)
        hour_sin = torch.sin(2 * math.pi * hour / 24.0).unsqueeze(1)  # [N, 1]
        hour_cos = torch.cos(2 * math.pi * hour / 24.0).unsqueeze(1)  # [N, 1]
        dow_sin = torch.sin(2 * math.pi * dow / 7.0).unsqueeze(1)  # [N, 1]
        dow_cos = torch.cos(2 * math.pi * dow / 7.0).unsqueeze(1)  # [N, 1]

        # Extract binary and contextual features
        alert_type = x[:, 1].unsqueeze(1)  # [N, 1]
        # Handle 5-dim input (no contextual features) by using zeros
        if x.size(1) >= 6 + self.n_contextual:
            contextual = x[:, 6:6 + self.n_contextual]  # [N, n_contextual]
        else:
            contextual = torch.zeros(n, self.n_contextual, device=x.device)  # [N, n_contextual]

        # Concatenate all features
        combined = torch.cat([
            tactic_emb,
            protocol_emb,
            service_emb,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            alert_type,
            contextual,
        ], dim=1)  # [N, total_input_dim]

        # Project to hidden_dim
        output = self.projection(combined)  # [N, hidden_dim]

        return output

    def get_vocab_info(self) -> dict:
        """Return vocabulary sizes for checkpoint saving."""
        return {
            'tactic': self.tactic_embed.num_embeddings,
            'protocol': self.protocol_embed.num_embeddings,
            'service': self.service_embed.num_embeddings,
        }


def compute_vocab_sizes(datasets: list, dataset_paths: dict) -> dict:
    """
    Compute vocabulary sizes for categorical features across all datasets.

    Args:
        datasets: List of dataset names (e.g., ['unsw_nb15', 'nsl_kdd', 'ton_iot'])
        dataset_paths: Dict mapping dataset name to path pattern
            (e.g., {'unsw_nb15': 'datasets/unsw_nb15/mitre_format.csv'})

    Returns:
        Dict with max vocab sizes for tactic, protocol, service
    """
    import pandas as pd
    from pathlib import Path

    max_tactic = 1  # 0 is padding/unknown
    max_protocol = 1
    max_service = 1

    for ds_name in datasets:
        path = Path(dataset_paths.get(ds_name, f"datasets/{ds_name}/mitre_format.csv"))

        # Try parquet first, then csv
        parquet_path = path.with_suffix('.parquet')
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif path.exists():
            df = pd.read_csv(path)
        else:
            continue

        if 'tactic' in df.columns:
            unique_tactics = df['tactic'].nunique()
            max_tactic = max(max_tactic, unique_tactics + 1)  # +1 for padding

        if 'protocol' in df.columns:
            unique_protocols = df['protocol'].nunique()
            max_protocol = max(max_protocol, unique_protocols + 1)

        if 'service' in df.columns:
            unique_services = df['service'].nunique()
            max_service = max(max_service, unique_services + 1)

    # Add margin for unseen categories at inference
    return {
        'tactic': min(max_tactic + 10, 100),  # Cap at 100
        'protocol': min(max_protocol + 50, 300),  # Cap at 300
        'service': min(max_service + 10, 100),  # Cap at 100
    }
