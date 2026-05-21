"""
MITRE-CORE HGNN Module
=======================
Heterogeneous Graph Neural Network for Advanced Alert Correlation.

Replaces Union-Find correlation with learned graph embeddings.

Architecture
------------
- Heterogeneous Graph Attention Network (GATConv via PyG HeteroConv)
- Multi-relation edges: shares_ip, shares_host, temporal_near,
  user_to_alert, host_to_alert, and IoT / Linux-APT variants
- Learns optimal feature weights automatically vs. handcrafted 0.6/0.3/0.1
- O(n+e) complexity vs. O(n²) for Union-Find

Changelog (v2.1 — Adaptive Confidence Integration)
----------------------------------------------------
  - Updated: HGNNCorrelationEngine.correlate() now feeds ``cluster_confidence``
    scores back into enhanced_correlation() via a confidence-gated UF fallback
    pass for alerts the HGNN is uncertain about (below CONFIDENCE_GATE).
  - New constant: CONFIDENCE_GATE (default 0.6) — alerts with max softmax
    probability below this value are re-correlated with the UF engine using
    confidence_guided_threshold() as the threshold driver.
  - New method: HGNNCorrelationEngine._uf_refinement_pass() — isolated,
    testable method encapsulating the low-confidence UF re-pass logic.
  - No changes to MITREHeteroGNN, AlertToGraphConverter, or training components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATConv, Linear, global_mean_pool
from torch_geometric.data import HeteroData
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from pathlib import Path

logger = logging.getLogger("mitre-core.hgnn")

# ---------------------------------------------------------------------------
# Module-level constant — confidence gate for UF refinement pass
# ---------------------------------------------------------------------------

#: Alerts with HGNN max-softmax confidence below this value trigger a
#: Union-Find re-correlation pass driven by confidence_guided_threshold().
#: 0.6 was chosen based on the sensitivity study: threshold ≥ 0.7 pushed
#: ARI to 0.9708, and calibration results show ECE degrades most for alerts
#: the HGNN scores below 0.6.  Tune via HGNNCorrelationEngine(confidence_gate=…).
_DEFAULT_CONFIDENCE_GATE: float = 0.6


def soft_zca_whiten(embeddings: np.ndarray, eps: float = 0.1) -> np.ndarray:
    """Soft-ZCA whitening to decorrelate collapsed GNN embeddings.
    
    Applies W = U(Λ + εI)^{-1/2} U^T to the centered embedding matrix.
    eps controls regularization: larger eps = less aggressive whitening.
    From arXiv:2411.17538 — fixes cosine_sim > 0.95 collapse without retraining.
    """
    X = embeddings - embeddings.mean(axis=0, keepdims=True)
    cov = (X.T @ X) / max(len(X) - 1, 1)
    U, S, _ = np.linalg.svd(cov, full_matrices=False)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    whitened = X @ W.T
    # Re-normalize to unit sphere after whitening
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return whitened / norms


# ============================================================================
# Homogeneous GNN Baseline
# ============================================================================

class HomogeneousGNN(nn.Module):
    """
    Homogeneous GNN Baseline (GCN) for comparison against HGNN.
    Treats all nodes as 'alerts' and projects entity features into alert space
    or creates generic edges between alerts based on shared entities.
    """

    def __init__(
        self,
        input_dim: int = 8,
        feature_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_clusters: int = 10,
    ):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.encoder = nn.Linear(input_dim, feature_dim)
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = feature_dim if i == 0 else hidden_dim
            self.convs.append(GCNConv(in_dim, hidden_dim))
        self.dropout = dropout
        self.cluster_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_clusters),
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        cluster_logits = self.cluster_classifier(x)
        return cluster_logits, x


# ============================================================================
# MITREHeteroGNN — core heterogeneous GNN model (unchanged from v2.0)
# ============================================================================

class MITREHeteroGNN(nn.Module):
    """
    Heterogeneous Graph Neural Network for MITRE-CORE.

    Node Types
    ----------
    alert       : Security alerts (main entity)
    user        : Source/destination users
    host        : Source/destination/device hosts
    ip          : IP addresses (source/destination/device)
    device      : IIoT devices (derived from ports)
    gateway     : Network gateways (derived from subnets)
    process     : Linux process names (Linux-APT datasets)
    command_line: Command-line strings (Linux-APT datasets)

    Edge Types
    ----------
    alert-shares_ip-alert       : Alerts sharing IP addresses
    alert-shares_host-alert     : Alerts sharing hostnames
    alert-temporal_near-alert   : Alerts within time window
    user-owns-alert             : User associated with alert
    host-generates-alert        : Host associated with alert
    ip-involved_in-alert        : IP involved in alert
    device-connects_via-gateway : Device connects via gateway
    sensor_type-classifies-device: Device type classification
    process-executes-alert      : Process associated with alert (APT)
    command_line-associated_with-alert: Command line (APT)
    """

    def __init__(
        self,
        alert_feature_dim: int = 6,
        user_feature_dim: int = 32,
        host_feature_dim: int = 32,
        ip_feature_dim: int = 32,
        device_feature_dim: int = 32,
        gateway_feature_dim: int = 16,
        process_feature_dim: int = 32,
        command_line_feature_dim: int = 64,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_clusters: int = 10,
        domain_cluster_dims: Optional[Dict[str, int]] = None,
        vocab_sizes: Optional[Dict[str, int]] = None,
        aggr_method: str = "mean",
    ):
        super().__init__()

        self.alert_feature_dim = alert_feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.aggr_method = aggr_method

        # Input-side residual projection (B1: preserves raw feature variance)
        self.alert_raw_proj = Linear(alert_feature_dim, hidden_dim)

        # Input projections
        # Use CategoricalAlertEncoder if vocab_sizes provided, else fallback to Linear
        if vocab_sizes is not None:
            from hgnn.categorical_encoder import CategoricalAlertEncoder
            self.alert_encoder = CategoricalAlertEncoder(
                hidden_dim=hidden_dim,
                n_contextual=0,  # B1: 6-dim base features, no contextual
                n_tactics=vocab_sizes.get('tactic', 50),
                n_protocols=vocab_sizes.get('protocol', 200),
                n_services=vocab_sizes.get('service', 50),
            )
        else:
            # Backward compatibility: plain Linear for old checkpoints
            self.alert_encoder = Linear(-1, hidden_dim)
        self.user_encoder = Linear(-1, hidden_dim)
        self.host_encoder = Linear(-1, hidden_dim)
        self.ip_encoder = Linear(-1, hidden_dim)
        self.device_encoder = Linear(-1, hidden_dim)
        self.gateway_encoder = Linear(-1, hidden_dim)
        self.sensor_type_encoder = Linear(-1, hidden_dim)
        self.source_sensor_encoder = Linear(-1, hidden_dim)  # NEW: origin sensor node (CS-2)
        self.process_encoder = Linear(-1, hidden_dim)
        self.command_line_encoder = Linear(-1, hidden_dim)
        self.container_encoder = Linear(-1, hidden_dim)
        self.pod_encoder = Linear(-1, hidden_dim)

        # Heterogeneous GNN layers
        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            conv_dict = {}

            # Alert-to-Alert (intra-type)
            for rel in ("shares_ip", "shares_host", "temporal_near", "semantic_similar", "precedes"):
                conv_dict[("alert", rel, "alert")] = GATConv(
                    hidden_dim, hidden_dim // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False,
                )

            # Cross-type bidirectional edges
            cross_edges = [
                ("user", "owns", "alert"),
                ("alert", "owned_by", "user"),
                ("host", "generates", "alert"),
                ("alert", "generated_by", "host"),
                ("ip", "involved_in", "alert"),
                ("alert", "involves", "ip"),
                ("device", "connects_via", "gateway"),
                ("gateway", "connected_to", "device"),
                ("sensor_type", "classifies", "device"),
                ("device", "classified_as", "sensor_type"),
                ("device", "generates", "alert"),
                ("alert", "generated_by", "device"),
                ("process", "executes", "alert"),
                ("alert", "executed_by", "process"),
                ("command_line", "associated_with", "alert"),
                ("alert", "has", "command_line"),
                ("container", "runs_in", "pod"),
                ("pod", "runs", "container"),
                ("process", "spawned_in", "container"),
                ("container", "spawns", "process"),
                ("ip", "resolves_to", "host"),       # bridge edge
                ("host", "resolved_from", "ip"),     # bridge edge (reverse)
                ("alert", "collected_by", "source_sensor"),    # NEW (CS-2)
                ("source_sensor", "collects", "alert"),        # NEW (reverse)
            ]
            for edge_type in cross_edges:
                conv_dict[edge_type] = GATConv(
                    hidden_dim, hidden_dim // num_heads,
                    heads=num_heads, dropout=dropout, add_self_loops=False,
                )

            self.convs.append(HeteroConv(conv_dict, aggr=self.aggr_method))

        # LayerNorm per layer to prevent over-smoothing (Change B)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

        # Cluster classification head
        self.cluster_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_clusters),
        )

        self.attention_weights: Dict = {}

    def forward(
        self, data: HeteroData, domain: Optional[str] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through HGNN.

        Args
        ----
        data : HeteroData
            Graph with node features and edge indices.

        Returns
        -------
        cluster_logits : torch.Tensor [num_alerts, num_clusters]
        node_embeddings : Dict[str, torch.Tensor]
            Per-node-type embedding tensors.
        """
        node_types = data.node_types
        encoder_map = {
            "alert": self.alert_encoder,
            "user": self.user_encoder,
            "host": self.host_encoder,
            "ip": self.ip_encoder,
            "device": self.device_encoder,
            "gateway": self.gateway_encoder,
            "sensor_type": self.sensor_type_encoder,
            "source_sensor": self.source_sensor_encoder,
            "process": self.process_encoder,
            "command_line": self.command_line_encoder,
            "container": self.container_encoder,
            "pod": self.pod_encoder,
        }

        # B1: Capture raw alert features for input-side residual
        alert_raw = data["alert"].x if "alert" in node_types else None
        
        x_dict: Dict[str, torch.Tensor] = {}
        for ntype, encoder in encoder_map.items():
            if ntype in node_types:
                x_dict[ntype] = encoder(data[ntype].x)

        if "alert" not in x_dict:
            raise ValueError("Data must contain 'alert' nodes.")

        # Guard against empty edge_index_dict
        if not hasattr(data, 'edge_index_dict') or not data.edge_index_dict:
            # Return zero embeddings if no edges
            return {nt: torch.zeros(x_dict[nt].size(0), self.hidden_dim, device=x_dict[nt].device) 
                   for nt in x_dict.keys()}
        
        # Filter edges to available node types
        available_edges = {
            et: ei
            for et, ei in data.edge_index_dict.items()
            if et[0] in x_dict and et[2] in x_dict
        }

        for i, conv in enumerate(self.convs):
            conv_edges = {et: available_edges[et] for et in conv.convs if et in available_edges}
            if not conv_edges:
                continue
            
            # Message passing
            x_dict_new = conv(x_dict, conv_edges)
            
            # LayerNorm + ReLU + Dropout + Residual (Change B)
            norm = self.layer_norms[i]
            for k, v in x_dict_new.items():
                # Residual connection: new = norm(dropout(relu(conv(x)))) + x
                residual = x_dict.get(k, torch.zeros_like(v))
                v = F.relu(v)
                v = F.dropout(v, p=0.3, training=self.training)
                v = norm(v)
                x_dict[k] = v + residual  # Skip connection

        # B1: Input-side residual skip for alert embeddings (preserves raw feature variance)
        if (alert_raw is not None and "alert" in x_dict and 
            hasattr(self, 'alert_raw_proj') and 
            alert_raw.shape[1] == self.alert_feature_dim):
            alert_residual = self.alert_raw_proj(alert_raw)
            x_dict["alert"] = x_dict["alert"] + alert_residual

        alert_embeddings = x_dict["alert"]
        cluster_logits = self.cluster_classifier(alert_embeddings)
        return cluster_logits, x_dict

    def get_backbone_embeddings(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """
        Extract backbone embeddings (before classification head) for explainability.
        
        Args:
            data: HeteroData graph
            
        Returns:
            Dictionary of node embeddings per node type
        """
        self.eval()
        with torch.no_grad():
            node_types = data.node_types
            encoder_map = {
                "alert": self.alert_encoder,
                "user": self.user_encoder,
                "host": self.host_encoder,
                "ip": self.ip_encoder,
                "device": self.device_encoder,
                "gateway": self.gateway_encoder,
                "sensor_type": self.sensor_type_encoder,
            "source_sensor": self.source_sensor_encoder,
            "process": self.process_encoder,
            "command_line": self.command_line_encoder,
            "container": self.container_encoder,
            "pod": self.pod_encoder,
            }

            x_dict: Dict[str, torch.Tensor] = {}
            for ntype, encoder in encoder_map.items():
                if ntype in node_types:
                    x_dict[ntype] = encoder(data[ntype].x)

            if "alert" not in x_dict:
                raise ValueError("Data must contain 'alert' nodes.")

            # Filter edges to available node types
            available_edges = {
                et: ei
                for et, ei in data.edge_index_dict.items()
                if et[0] in x_dict and et[2] in x_dict
            }

            # Message passing layers (same as forward but without classification)
            for i, conv in enumerate(self.convs):
                conv_edges = {et: available_edges[et] for et in conv.convs if et in available_edges}
                if not conv_edges:
                    continue
                
                # Message passing
                x_dict_new = conv(x_dict, conv_edges)
                
                # LayerNorm + ReLU + Dropout + Residual
                norm = self.layer_norms[i]
                for k, v in x_dict_new.items():
                    residual = x_dict.get(k, torch.zeros_like(v))
                    v = F.relu(v)
                    v = F.dropout(v, p=0.3, training=self.training)
                    v = norm(v)
                    x_dict[k] = v + residual

            return x_dict

    def get_attention_weights(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        """Extract GAT attention weights for interpretability."""
        self.eval()
        attention_weights: Dict = {}
        with torch.no_grad():
            node_types = data.node_types
            encoder_map = {
                "alert": self.alert_encoder,
                "user": self.user_encoder,
                "host": self.host_encoder,
                "ip": self.ip_encoder,
                "device": self.device_encoder,
                "gateway": self.gateway_encoder,
                "sensor_type": self.sensor_type_encoder,
                "source_sensor": self.source_sensor_encoder,
                "process": self.process_encoder,
                "command_line": self.command_line_encoder,
                "container": self.container_encoder,
                "pod": self.pod_encoder,
            }
            x_dict = {
                ntype: enc(data[ntype].x)
                for ntype, enc in encoder_map.items()
                if ntype in node_types
            }
            for conv in self.convs:
                x_dict, attn = conv(x_dict, data.edge_index_dict, return_attention_weights=True)
                attention_weights.update(attn)
        return attention_weights


# ============================================================================
# AlertToGraphConverter — unchanged from v2.0
# ============================================================================

class AlertToGraphConverter:
    """
    Converts a MITRE-CORE alert DataFrame to PyTorch Geometric HeteroData.

    Handles node creation, edge construction, feature encoding, and temporal
    edge weighting. Compatible with all eight MITRE-CORE v2 datasets.
    """

    def __init__(self, temporal_window_hours: float = 1.0, entity_memory: Optional[nn.Module] = None,
                 build_bridge_edges: bool = True, collapse_entities: bool = False,
                 use_burstiness: bool = False, track_data_source: bool = False,
                 build_precedes_edges: bool = False, precedes_window_hours: float = 2.0):
        self.temporal_window = temporal_window_hours
        self.entity_memory = entity_memory
        self.build_bridge_edges = build_bridge_edges
        self.collapse_entities = collapse_entities
        self.use_burstiness = use_burstiness
        self.track_data_source = track_data_source
        self.build_precedes_edges = build_precedes_edges
        self.precedes_window_hours = precedes_window_hours

    def convert(self, df: pd.DataFrame) -> HeteroData:
        """Convert alert DataFrame to HeteroData."""
        data = HeteroData()

        if "AlertId" not in df.columns:
            df = df.copy()
            df["AlertId"] = [f"alert_{i}" for i in range(len(df))]

        alerts = df["AlertId"].unique()
        users = pd.concat([
            df["SourceUserName"].dropna() if "SourceUserName" in df.columns else pd.Series(dtype=str),
            df["DestinationUserName"].dropna() if "DestinationUserName" in df.columns else pd.Series(dtype=str),
        ]).unique()
        hosts = pd.concat([
            df["SourceHostName"].dropna() if "SourceHostName" in df.columns else pd.Series(dtype=str),
            df["DeviceHostName"].dropna() if "DeviceHostName" in df.columns else pd.Series(dtype=str),
            df["DestinationHostName"].dropna() if "DestinationHostName" in df.columns else pd.Series(dtype=str),
        ]).unique()
        ips = pd.concat([
            df["SourceAddress"].dropna() if "SourceAddress" in df.columns else pd.Series(dtype=str),
            df["DestinationAddress"].dropna() if "DestinationAddress" in df.columns else pd.Series(dtype=str),
            df["DeviceAddress"].dropna() if "DeviceAddress" in df.columns else pd.Series(dtype=str),
        ]).unique()

        gateways: set = set()
        devices: set = set()
        sensor_types: set = set()
        processes: set = set()
        command_lines: set = set()

        for _, row in df.iterrows():
            if all(c in df.columns for c in ("SourceUserName", "SourceAddress", "DeviceAddress")):
                u = str(row.get("SourceUserName", ""))
                if u.startswith("gateway_"):
                    gateways.add(u)
                    devices.add(str(row.get("SourceAddress", "")))
                    dev_addr = str(row.get("DeviceAddress", ""))
                    if ":" in dev_addr:
                        sensor_types.add(f"sensor_{dev_addr.split(':')[-1]}")
            if "ProcessName" in df.columns and pd.notna(row.get("ProcessName")):
                processes.add(str(row["ProcessName"]))
            if "CommandLine" in df.columns and pd.notna(row.get("CommandLine")):
                command_lines.add(str(row["CommandLine"]))

        gateways_l = list(gateways)
        devices_l = list(devices)
        sensor_types_l = list(sensor_types)
        processes_l = list(processes)
        command_lines_l = list(command_lines)

        alert_to_idx = {a: i for i, a in enumerate(alerts)}
        user_to_idx = {u: i for i, u in enumerate(users)}
        host_to_idx = {h: i for i, h in enumerate(hosts)}
        ip_to_idx = {ip: i for i, ip in enumerate(ips)}
        device_to_idx = {d: i for i, d in enumerate(devices_l)}
        gateway_to_idx = {g: i for i, g in enumerate(gateways_l)}
        sensor_type_to_idx = {s: i for i, s in enumerate(sensor_types_l)}
        process_to_idx = {p: i for i, p in enumerate(processes_l)}
        command_line_to_idx = {c: i for i, c in enumerate(command_lines_l)}

        features, ip_ids, host_ids, user_ids = self._encode_alert_features(df, user_to_idx, host_to_idx, ip_to_idx)
        alert_x = torch.tensor(features, dtype=torch.float)
        
        if self.entity_memory is not None:
            # Sort by timestamp (already done by user, but ensure order if needed)
            temporal_ctx = self.entity_memory(alert_x, ip_ids, host_ids, user_ids)
            # Residual connection with learned alpha (prevents noise)
            # Ensure dimensions match for residual connection
            if temporal_ctx.size(-1) == alert_x.size(-1):
                alert_x = alert_x + self.entity_memory.alpha * temporal_ctx
            else:
                # If dimensions don't match, use projection or concatenation
                logger.warning(f"Dimension mismatch: alert_x={alert_x.shape}, ctx={temporal_ctx.shape}")
                alert_x = torch.cat([alert_x, temporal_ctx], dim=1)
            
        data["alert"].x = alert_x

        for ntype, collection in [
            ("user", users), ("host", hosts), ("ip", ips),
            ("device", devices_l), ("gateway", gateways_l),
            ("sensor_type", sensor_types_l), ("process", processes_l),
            ("command_line", command_lines_l),
        ]:
            if len(collection) > 0:
                data[ntype].x = torch.ones(len(collection), 1)

        # NEW (CS-2): source_sensor nodes (only if track_data_source=True and column present)
        if self.track_data_source and "data_source" in df.columns:
            source_sensors = sorted(df["data_source"].dropna().unique().tolist())
            sensor_to_idx = {s: i for i, s in enumerate(source_sensors)}
            data["source_sensor"].x = torch.ones(len(source_sensors), 1)

        # Check if we should enforce label-pure edges (Fix 1 for SQTK_SIEM)
        # Edge filtering is only applied during training (when edge_labels is provided)
        # During inference, we use all edges for complete connectivity
        edge_labels = None  # Inference mode: no edge filtering

        edges = self._build_edges(
            df, alert_to_idx, user_to_idx, host_to_idx, ip_to_idx,
            device_to_idx, gateway_to_idx, sensor_type_to_idx,
            process_to_idx, command_line_to_idx,
            edge_labels=edge_labels
        )
        
        # Helper to add edges to the edges dict (initialize if missing)
        def add_edge(etype, src, dst):
            if etype not in edges:
                edges[etype] = ([], [])
            edges[etype][0].append(src)
            edges[etype][1].append(dst)
        
        # NEW (CS-2): Add collected_by edges for source_sensor
        if self.track_data_source and "data_source" in df.columns:
            for _, row in df.iterrows():
                src_name = row.get("data_source")
                if pd.notna(src_name) and src_name in sensor_to_idx:
                    aid = alert_to_idx[row["AlertId"]]
                    sid = sensor_to_idx[src_name]
                    add_edge(("alert", "collected_by", "source_sensor"), aid, sid)
                    add_edge(("source_sensor", "collects", "alert"), sid, aid)

        # NEW (CS-3): Kill-chain precedes edges
        # Connects last alert of each entity-group to the first alert of the next group
        if self.build_precedes_edges and "timestamp" in df.columns:
            try:
                df_sorted = df.sort_values("timestamp").reset_index(drop=True)
                ts = pd.to_datetime(df_sorted["timestamp"], errors="coerce")
                aidxs = [alert_to_idx[a] for a in df_sorted["AlertId"]]

                # Group by shared entity (src_ip or hostname) then connect across groups
                entity_last = {}  # entity -> (alert_idx, ts)
                for i, row in df_sorted.iterrows():
                    for col in ("src_ip", "hostname", "SourceAddress", "SourceHostName"):
                        ent = row.get(col)
                        if ent and pd.notna(ent) and str(ent).strip():
                            if ent in entity_last:
                                prev_idx, prev_ts = entity_last[ent]
                                diff_h = (ts.iloc[i] - prev_ts).total_seconds() / 3600
                                if 0 < diff_h <= self.precedes_window_hours:
                                    # directed: prev alert -> current alert
                                    add_edge(("alert", "precedes", "alert"), prev_idx, aidxs[i])
                            entity_last[ent] = (aidxs[i], ts.iloc[i])
                            break  # one entity match per alert is enough
            except Exception as e:
                logger.warning(f"precedes edge construction failed: {e}")

        # Ensure all expected edge types exist (even if empty) for model compatibility
        expected_edge_types = [
            ("alert", "shares_ip", "alert"),
            ("alert", "shares_host", "alert"),
            ("alert", "temporal_near", "alert"),
            ("user", "owns", "alert"),
            ("alert", "owned_by", "user"),
            ("host", "generates", "alert"),
            ("alert", "generated_by", "host"),
            ("ip", "involved_in", "alert"),
            ("alert", "involves", "ip"),
        ]
        # NEW (CS-2, CS-3): Add cross-sensor edge types if enabled
        if self.track_data_source and "data_source" in df.columns:
            expected_edge_types.extend([
                ("alert", "collected_by", "source_sensor"),
                ("source_sensor", "collects", "alert"),
            ])
        if self.build_precedes_edges:
            expected_edge_types.append(("alert", "precedes", "alert"))
        
        for edge_type in expected_edge_types:
            if edge_type not in edges:
                edges[edge_type] = ([], [])
        
        for edge_type, (src, dst) in edges.items():
            if src:
                data[edge_type].edge_index = torch.tensor([src, dst], dtype=torch.long)
            else:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        return data

    def _compute_contextual_features(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Compute graph-contextual features for each alert. These are NOT labels —
        they are observable statistics derived from the alert's neighborhood in
        the current batch. They provide discriminative signal even when raw
        alert features are degenerate.

        Returns: [N, 9] tensor of contextual features
        """
        import numpy as np
        n = len(df)
        feats = torch.zeros(n, 9)

        # Fixed denominator for consistency between train (chunks of ~500) and inference (2k-10k)
        denom = np.log1p(2000.0)

        # Tactic frequency
        if 'tactic' in df.columns:
            tactic_counts = df['tactic'].map(df['tactic'].value_counts())
            feats[:, 0] = torch.tensor(np.log1p(tactic_counts.fillna(0).values) / denom, dtype=torch.float32)
        elif 'Tactic' in df.columns:
            tactic_counts = df['Tactic'].map(df['Tactic'].value_counts())
            feats[:, 0] = torch.tensor(np.log1p(tactic_counts.fillna(0).values) / denom, dtype=torch.float32)

        # Service frequency
        if 'service' in df.columns:
            svc_counts = df['service'].map(df['service'].value_counts())
            feats[:, 1] = torch.tensor(np.log1p(svc_counts.fillna(0).values) / denom, dtype=torch.float32)

        # dst_ip frequency
        dst_col = 'dst_ip' if 'dst_ip' in df.columns else 'DestinationAddress' if 'DestinationAddress' in df.columns else None
        if dst_col:
            dst_counts = df[dst_col].map(df[dst_col].value_counts())
            feats[:, 2] = torch.tensor(np.log1p(dst_counts.fillna(0).values) / denom, dtype=torch.float32)

        # src_ip frequency
        src_col = 'src_ip' if 'src_ip' in df.columns else 'SourceAddress' if 'SourceAddress' in df.columns else None
        if src_col:
            src_counts = df[src_col].map(df[src_col].value_counts())
            feats[:, 3] = torch.tensor(np.log1p(src_counts.fillna(0).values) / denom, dtype=torch.float32)

        # Temporal density: O(n log n) with searchsorted
        time_col = None
        for col in ["timestamp", "Timestamp", "EndDate", "StartTime", "time", "Time"]:
            if col in df.columns:
                time_col = col
                break
                
        if time_col:
            times = pd.to_datetime(df[time_col], errors='coerce')
            valid = times.notna()
            if valid.sum() > 0:
                t_seconds = np.where(valid, times.values.astype('int64') / 1e9, np.nan)
                t_arr = t_seconds[valid]
                
                # Use searchsorted to avoid O(n^2) memory
                sort_idx = np.argsort(t_arr)
                t_sorted = t_arr[sort_idx]
                
                left_300 = np.searchsorted(t_sorted, t_sorted - 300, side='left')
                right_300 = np.searchsorted(t_sorted, t_sorted + 300, side='right')
                counts_300 = right_300 - left_300
                
                left_1800 = np.searchsorted(t_sorted, t_sorted - 1800, side='left')
                right_1800 = np.searchsorted(t_sorted, t_sorted + 1800, side='right')
                counts_1800 = right_1800 - left_1800
                
                left_7200 = np.searchsorted(t_sorted, t_sorted - 7200, side='left')
                right_7200 = np.searchsorted(t_sorted, t_sorted + 7200, side='right')
                counts_7200 = right_7200 - left_7200
                
                # Revert to original order
                unsort_idx = np.empty_like(sort_idx)
                unsort_idx[sort_idx] = np.arange(len(sort_idx))
                
                idx_valid = np.where(valid)[0]
                feats[idx_valid, 4] = torch.tensor(np.log1p(counts_300[unsort_idx]) / denom, dtype=torch.float32)
                feats[idx_valid, 5] = torch.tensor(np.log1p(counts_1800[unsort_idx]) / denom, dtype=torch.float32)
                feats[idx_valid, 6] = torch.tensor(np.log1p(counts_7200[unsort_idx]) / denom, dtype=torch.float32)

        # Alert index position - REMOVED because it's purely noise when not grouped by campaign
        feats[:, 7] = 0.0

        # Protocol × service interaction
        if 'protocol' in df.columns and 'service' in df.columns:
            combo = df['protocol'].astype(str) + '_' + df['service'].astype(str)
            combo_counts = combo.map(combo.value_counts())
            feats[:, 8] = torch.tensor(np.log1p(combo_counts.fillna(0).values) / denom, dtype=torch.float32)

        return feats

    def _encode_alert_features(
        self, df: pd.DataFrame, user_to_idx: Dict[str, int] = None, host_to_idx: Dict[str, int] = None, ip_to_idx: Dict[str, int] = None
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tactic
        tactics = pd.Categorical(df["tactic"]).codes if "tactic" in df.columns else np.zeros(len(df))
        if "Tactic" in df.columns and "tactic" not in df.columns:
            tactics = pd.Categorical(df["Tactic"]).codes

        # Alert Type
        if "alert_type" in df.columns:
            alert_types = (df["alert_type"] == "attack").astype(int).values
        elif "AttackTechnique" in df.columns:
            alert_types = (df["AttackTechnique"] != "").astype(int).values
        else:
            alert_types = np.zeros(len(df))

        # Temporal
        try:
            # Try multiple timestamp column names (case-insensitive)
            ts_col = None
            for col in ["timestamp", "Timestamp", "EndDate", "StartTime", "time", "Time"]:
                if col in df.columns:
                    ts_col = col
                    break
            
            if ts_col is None:
                raise ValueError("No timestamp column found")
            
            ts_values = df[ts_col]
            
            # Handle Unix timestamps (integers/floats) vs datetime strings
            if pd.api.types.is_numeric_dtype(ts_values):
                # Unix timestamps - convert to datetime
                dates = pd.to_datetime(ts_values, unit='s', errors='coerce')
            else:
                # Datetime strings
                dates = pd.to_datetime(ts_values, errors='coerce')
            
            hour = np.nan_to_num(dates.dt.hour.values, nan=0.0)
            dow = np.nan_to_num(dates.dt.dayofweek.values, nan=0.0)
        except (ValueError, TypeError, AttributeError):
            hour = np.zeros(len(df))
            dow = np.zeros(len(df))

        # Protocol
        protocols = pd.Categorical(df["protocol"]).codes if "protocol" in df.columns else np.zeros(len(df))

        # Service
        services = pd.Categorical(df["service"]).codes if "service" in df.columns else np.zeros(len(df))

        # New Enriched Features (9 dims) for SQTK_SIEM -> expanded to 14 dims for B1/B3
        # 1-2. Bytes (log normalized)
        src_bytes = np.log1p(df["src_bytes"].fillna(0).astype(float).values) if "src_bytes" in df.columns else np.zeros(len(df))
        dst_bytes = np.log1p(df["dst_bytes"].fillna(0).astype(float).values) if "dst_bytes" in df.columns else np.zeros(len(df))
        
        # 3. Severity (categorical encoding for string values)
        if "severity" in df.columns:
            # Handle string severity levels (Critical, High, Medium, Low)
            severity = pd.Categorical(df["severity"].fillna("Unknown")).codes
        else:
            severity = np.zeros(len(df))
            
        # 4-5. Ports (Binned: 0=<1024, 1=1024-49151, 2=>=49152)
        def bin_port(p):
            if pd.isna(p) or str(p).strip().upper() in ['NIL', 'NULL', '', 'NONE', 'N/A', 'NA', 'UNKNOWN', '-']:
                return 0
            try:
                p_val = float(p)
                if p_val < 1024: return 0
                if p_val < 49152: return 1
                return 2
            except (ValueError, TypeError):
                return 0

        src_port = df["src_port"].apply(bin_port).values if "src_port" in df.columns else np.zeros(len(df))
        dst_port = df["dst_port"].apply(bin_port).values if "dst_port" in df.columns else np.zeros(len(df))

        # 6-8. Additional Categoricals
        stages = pd.Categorical(df["stage"]).codes if "stage" in df.columns else np.zeros(len(df))
        device_types = pd.Categorical(df["device_type"]).codes if "device_type" in df.columns else np.zeros(len(df))
        techniques = pd.Categorical(df["technique"]).codes if "technique" in df.columns else np.zeros(len(df))
        
        # 9. Campaign ID (if available, used for clustering hints in some setups, else 0)
        campaigns = pd.Categorical(df["campaign_id"]).codes if "campaign_id" in df.columns else np.zeros(len(df))
        
        # Track B1: Temporal Burstiness (rolling 1s, 10s, 60s counts per src_ip)
        burst_1s = np.zeros(len(df))
        burst_10s = np.zeros(len(df))
        burst_60s = np.zeros(len(df))
        
        if ts_col is not None and "src_ip" in df.columns:
            # Use pandas rolling window on timestamp per src_ip
            df_temp = df[[ts_col, "src_ip"]].copy()
            df_temp['_idx'] = np.arange(len(df))
            df_temp['_dummy'] = 1
            if pd.api.types.is_numeric_dtype(df_temp[ts_col]):
                df_temp['ts'] = pd.to_datetime(df_temp[ts_col], unit='s')
            else:
                df_temp['ts'] = pd.to_datetime(df_temp[ts_col], errors='coerce')
                
            df_temp = df_temp.sort_values(by=['src_ip', 'ts'])
            
            df_indexed = df_temp.set_index('ts')
            
            for window_str, out_arr in [('1s', burst_1s), ('10s', burst_10s), ('60s', burst_60s)]:
                try:
                    df_sorted = df_temp.sort_values(by=['src_ip', 'ts']).set_index('ts')
                    counts = df_sorted.groupby('src_ip')['_dummy'].rolling(window_str).count().values
                    
                    original_indices = df_temp.sort_values(by=['src_ip', 'ts'])['_idx'].values
                    # Log-normalize the counts
                    out_arr[original_indices] = np.log1p(counts)
                except Exception as e:
                    logger.warning(f"Failed to compute temporal burstiness {window_str}: {e}")
                    
        # Track B3: Flow-tail features (p95 of sbytes/dbytes per 5-alert window per IP)
        tail_sbytes = np.zeros(len(df))
        tail_dbytes = np.zeros(len(df))
        
        if "src_ip" in df.columns and "src_bytes" in df.columns and "dst_bytes" in df.columns:
            try:
                df_tail = df[["src_ip", "src_bytes", "dst_bytes"]].copy()
                df_tail['_idx'] = np.arange(len(df))
                df_tail['src_bytes'] = pd.to_numeric(df_tail['src_bytes'], errors='coerce').fillna(0)
                df_tail['dst_bytes'] = pd.to_numeric(df_tail['dst_bytes'], errors='coerce').fillna(0)
                
                # Sort by timestamp (using ts_col if available, else just index)
                if ts_col is not None:
                    df_tail[ts_col] = df[ts_col]
                    df_tail = df_tail.sort_values(by=['src_ip', ts_col])
                else:
                    df_tail = df_tail.sort_values(by=['src_ip', '_idx'])
                
                # Rolling 5-alert window p95
                grouped = df_tail.groupby('src_ip')
                sbytes_p95 = grouped['src_bytes'].rolling(window=5, min_periods=1).quantile(0.95).values
                dbytes_p95 = grouped['dst_bytes'].rolling(window=5, min_periods=1).quantile(0.95).values
                
                original_indices = df_tail['_idx'].values
                tail_sbytes[original_indices] = np.log1p(sbytes_p95)
                tail_dbytes[original_indices] = np.log1p(dbytes_p95)
            except Exception as e:
                logger.warning(f"Failed to compute flow-tail features: {e}")

        # NEW (CS-1): Add source_sensor_id as dimension 21 (guarded — zero if column absent)
        source_ids = (
            pd.Categorical(df["data_source"]).codes.astype(float)
            if (self.track_data_source and "data_source" in df.columns)
            else np.zeros(len(df))
        )

        # Output raw values for temporal (cyclical encoding in model)
        # and raw integer codes for categoricals (embedding lookup in model)
        # Total dims: 6 original + 9 new + 5 track_b + 1 source = 21 dims
        features = np.column_stack([
            tactics, alert_types, hour, dow, protocols, services,
            src_bytes, dst_bytes, severity, src_port, dst_port,
            stages, device_types, techniques, campaigns,
            burst_1s, burst_10s, burst_60s, tail_sbytes, tail_dbytes,
            source_ids,  # dim 21: data source provenance (0 if not tracked)
        ])
        features = np.nan_to_num(features, nan=0.0)
        
        # Create entity ID tensors for TGN memory
        n_alerts = len(df)
        ip_ids = torch.zeros(n_alerts, dtype=torch.long)
        host_ids = torch.zeros(n_alerts, dtype=torch.long)
        user_ids = torch.zeros(n_alerts, dtype=torch.long)
        
        if ip_to_idx is not None:
            for i, row in df.iterrows():
                for col in ("SourceAddress", "DestinationAddress", "DeviceAddress"):
                    if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                        ip_ids[i] = ip_to_idx.get(str(row[col]), 0)
                        break
        
        if host_to_idx is not None:
            for i, row in df.iterrows():
                for col in ("SourceHostName", "DeviceHostName", "DestinationHostName"):
                    if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                        host_ids[i] = host_to_idx.get(str(row[col]), 0)
                        break
        
        if user_to_idx is not None:
            for i, row in df.iterrows():
                for col in ("SourceUserName", "DestinationUserName"):
                    if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
                        user_ids[i] = user_to_idx.get(str(row[col]), 0)
                        break
        
        return features, ip_ids, host_ids, user_ids

    def _build_edges(
        self, df, alert_to_idx, user_to_idx, host_to_idx, ip_to_idx,
        device_to_idx, gateway_to_idx, sensor_type_to_idx,
        process_to_idx, command_line_to_idx,
        edge_labels: Optional[np.ndarray] = None,
    ) -> Dict:
        edges: Dict = defaultdict(lambda: ([], []))

        def add_edge(etype, src, dst):
            # If edge_labels is provided (training time), drop cross-label edges
            # to prevent GCN over-smoothing from actively fighting SupCon separation.
            if edge_labels is not None:
                if edge_labels[src] != edge_labels[dst]:
                    return
            edges[etype][0].append(src)
            edges[etype][1].append(dst)

        _NULL_SENTINELS = {"nil", "null", "none", "n/a", "na", "unknown", "-", ""}
        def _is_valid(val):
            return pd.notna(val) and str(val).strip().lower() not in _NULL_SENTINELS

        # Alert-to-Alert via shared IPs
        ip_to_alerts: Dict = defaultdict(list)
        for _, row in df.iterrows():
            aid = alert_to_idx[row["AlertId"]]
            for col in ("SourceAddress", "DestinationAddress", "DeviceAddress"):
                if col in row and _is_valid(row[col]):
                    ip_to_alerts[row[col]].append(aid)
        for _, idxs in ip_to_alerts.items():
            for i, ai in enumerate(idxs):
                # CAP EDGES: max 5 to prevent over-smoothing (arXiv:2403.09118)
                for aj in idxs[i + 1:i + 6]:
                    add_edge(("alert", "shares_ip", "alert"), ai, aj)
                    add_edge(("alert", "shares_ip", "alert"), aj, ai)

        # Alert-to-Alert via shared hosts
        host_to_alerts: Dict = defaultdict(list)
        for _, row in df.iterrows():
            aid = alert_to_idx[row["AlertId"]]
            for col in ("SourceHostName", "DeviceHostName", "DestinationHostName"):
                if col in row and _is_valid(row[col]):
                    host_to_alerts[row[col]].append(aid)
        for _, idxs in host_to_alerts.items():
            for i, ai in enumerate(idxs):
                # CAP EDGES: max 5 to prevent over-smoothing
                for aj in idxs[i + 1:i + 6]:
                    add_edge(("alert", "shares_host", "alert"), ai, aj)
                    add_edge(("alert", "shares_host", "alert"), aj, ai)

        # Temporal edges
        if "EndDate" in df.columns:
            try:
                df_s = df.sort_values("EndDate")
                ts = pd.to_datetime(df_s["EndDate"])
                aidxs = [alert_to_idx[a] for a in df_s["AlertId"]]
                for i, (ai, tsi) in enumerate(zip(aidxs, ts)):
                    edges_added = 0
                    for j in range(i + 1, min(i + 100, len(aidxs))):
                        if edges_added >= 5:  # CAP EDGES: max 5
                            break
                        diff_h = abs((ts.iloc[j] - tsi).total_seconds() / 3600)
                        if diff_h <= self.temporal_window:
                            add_edge(("alert", "temporal_near", "alert"), ai, aidxs[j])
                            add_edge(("alert", "temporal_near", "alert"), aidxs[j], ai)
                            edges_added += 1
                        else:
                            break
            except (ValueError, TypeError, KeyError):
                pass

        # Cross-type edges
        ip_to_host: Dict[str, str] = {}  # Mine IP→hostname mapping from alerts with both
        _NULL_SENTINELS = {"nil", "null", "none", "n/a", "na", "unknown", "-", ""}
        
        for _, row in df.iterrows():
            aid = alert_to_idx[row["AlertId"]]

            # Mine IP→host mapping from alerts that have both IP and hostname
            for ip_col, host_col in [
                ("SourceAddress", "SourceHostName"),
                ("DestinationAddress", "DestinationHostName"),
                ("DeviceAddress", "DeviceHostName"),
            ]:
                ip = row.get(ip_col)
                hst = row.get(host_col)
                if (pd.notna(ip) and pd.notna(hst) and
                    str(ip).strip().lower() not in _NULL_SENTINELS and
                    str(hst).strip().lower() not in _NULL_SENTINELS):
                    ip_to_host[str(ip)] = str(hst)
        
        for _, row in df.iterrows():
            aid = alert_to_idx[row["AlertId"]]

            if "SourceUserName" in row and pd.notna(row["SourceUserName"]) and row["SourceUserName"] in user_to_idx:
                uid = user_to_idx[row["SourceUserName"]]
                add_edge(("user", "owns", "alert"), uid, aid)
                add_edge(("alert", "owned_by", "user"), aid, uid)

            for col in ("SourceHostName", "DeviceHostName", "DestinationHostName"):
                if col in row and pd.notna(row[col]) and row[col] in host_to_idx:
                    hid = host_to_idx[row[col]]
                    add_edge(("host", "generates", "alert"), hid, aid)
                    add_edge(("alert", "generated_by", "host"), aid, hid)

            for col in ("SourceAddress", "DestinationAddress", "DeviceAddress"):
                if col in row and pd.notna(row[col]) and row[col] in ip_to_idx:
                    iid = ip_to_idx[row[col]]
                    add_edge(("ip", "involved_in", "alert"), iid, aid)
                    add_edge(("alert", "involves", "ip"), aid, iid)
                    
                    # Entity collapse: also connect alert to resolved host node
                    if self.collapse_entities and row[col] in ip_to_host:
                        host_val = ip_to_host[row[col]]
                        if host_val in host_to_idx:
                            hid = host_to_idx[host_val]
                            add_edge(("host", "generates", "alert"), hid, aid)
                            add_edge(("alert", "generated_by", "host"), aid, hid)

            u = str(row.get("SourceUserName", ""))
            src_ip = str(row.get("SourceAddress", ""))
            dev_addr = str(row.get("DeviceAddress", ""))
            if u.startswith("gateway_") and src_ip in device_to_idx and u in gateway_to_idx:
                did = device_to_idx[src_ip]
                gid = gateway_to_idx[u]
                add_edge(("device", "connects_via", "gateway"), did, gid)
                add_edge(("gateway", "connected_to", "device"), gid, did)
                add_edge(("device", "generates", "alert"), did, aid)
                add_edge(("alert", "generated_by", "device"), aid, did)
                if ":" in dev_addr:
                    st = f"sensor_{dev_addr.split(':')[-1]}"
                    if st in sensor_type_to_idx:
                        sid = sensor_type_to_idx[st]
                        add_edge(("sensor_type", "classifies", "device"), sid, did)
                        add_edge(("device", "classified_as", "sensor_type"), did, sid)

            if "ProcessName" in df.columns and pd.notna(row.get("ProcessName")) and row["ProcessName"] in process_to_idx:
                pid = process_to_idx[row["ProcessName"]]
                add_edge(("process", "executes", "alert"), pid, aid)
                add_edge(("alert", "executed_by", "process"), aid, pid)

            if "CommandLine" in df.columns and pd.notna(row.get("CommandLine")) and row["CommandLine"] in command_line_to_idx:
                cid = command_line_to_idx[row["CommandLine"]]
                add_edge(("command_line", "associated_with", "alert"), cid, aid)
                add_edge(("alert", "has", "command_line"), aid, cid)

        # Cross-sensor bridge edges: IP resolves to hostname (mined from alerts with both)
        if self.build_bridge_edges:
            logger.info(f"Building bridge edges: found {len(ip_to_host)} IP->hostname mappings")
            bridge_edges_added = 0
            for ip_val, host_val in ip_to_host.items():
                if ip_val in ip_to_idx and host_val in host_to_idx:
                    iid = ip_to_idx[ip_val]
                    hid = host_to_idx[host_val]
                    add_edge(("ip", "___resolves_to___", "host"), iid, hid)
                    add_edge(("host", "___resolved_from___", "ip"), hid, iid)
                    bridge_edges_added += 1
            logger.info(f"Added {bridge_edges_added} bridge edges (IP->hostname)")
        else:
            logger.info("Bridge edge construction DISABLED")

        return dict(edges)
# EmbeddingConfidenceScorer — Geometry-Aware Confidence (v2.2)

class EmbeddingConfidenceScorer:
    """
    Geometry-Aware Embedding Confidence (GAEC) v2 scorer.

    v2 changes from v1:
    - Replaced k-means (requires fixed n_centroids) with HDBSCAN (discovers
      cluster count automatically, handles noise, produces native probabilities).
    - Added PCA whitening pre-processing to amplify geometric variance in
      over-smoothed GNN embeddings before clustering.

    Over-smoothing diagnosis:
        If std(confidence) < 0.01 on any dataset, GNN embeddings have collapsed.
        This is diagnosed by checking mean pairwise cosine similarity of the
        raw embeddings — if > 0.95, over-smoothing is confirmed and num_layers
        should be reduced further or a residual skip connection added.

    Parameters
    ----------
    min_cluster_size : int
        Minimum number of alerts to form a cluster in HDBSCAN.
        Default 5 is conservative — appropriate for small sampled batches.
        For large batches (>5000 alerts), consider 20–50.
    min_samples : int
        HDBSCAN robustness parameter. Higher = more conservative clustering,
        more noise points. Default 3.
    pca_components : int or None
        Number of PCA components to retain before HDBSCAN. None = no PCA.
        Default 32 — retains meaningful variance while removing noise dims.
    metric : str
        Distance metric for HDBSCAN. 'cosine' is appropriate for L2-normalised
        GNN embeddings. 'euclidean' works for non-normalised.
    fallback_to_uniform : bool
        If HDBSCAN finds 0 or 1 cluster (all noise), return uniform confidence
        of 0.5 rather than crashing. This triggers maximum UF routing, which
        is the correct behaviour when the HGNN has no geometric structure.
    noise_point_strategy : str
        How to handle HDBSCAN noise points (probability=0.0) in the GAEC scorer.
        "zero"        — keep confidence=0.0 (v2.5 behavior, routes to UF).
        "soft_assign" — assign nearest-neighbor cosine confidence in [0.05, 0.4].
        Only relevant when use_geometric_confidence=True.
    """

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        pca_components: Optional[int] = 32,
        metric: str = "cosine",
        fallback_to_uniform: bool = True,
        noise_point_strategy: str = "zero",
        auto_tune: bool = False,
        cluster_selection_epsilon: float = 0.0,
        use_umap: bool = False,
        umap_n_components: int = 10,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        cluster_selection_method: str = "eom",
        seed: int = 42,
        hdbscan_metric_fallback: bool = False,
        use_zca_whitening: bool = False,
        zca_eps: float = 0.1,
        clustering_method: str = "hdbscan",
        n_clusters: Optional[int] = None,
        bgmm_n_components: int = 30,
        prototype_checkpoint_path: Optional[str] = None,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.pca_components = pca_components
        self.metric = metric
        self.fallback_to_uniform = fallback_to_uniform
        self.noise_point_strategy = noise_point_strategy
        self.auto_tune = auto_tune
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.use_umap = use_umap
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.cluster_selection_method = cluster_selection_method
        self.hdbscan_metric_fallback = hdbscan_metric_fallback
        self.use_zca_whitening = use_zca_whitening
        self.zca_eps = zca_eps
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.bgmm_n_components = bgmm_n_components
        self.prototype_checkpoint_path = prototype_checkpoint_path
        self.seed = seed
        self._pca = None
        self._clusterer = None

    def fit_score(self, embeddings: torch.Tensor, confidence_gate: float = 0.6, raw_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fit HDBSCAN on embeddings and return per-alert confidence scores.
        """
        # Check if we should use fallback confidence scoring
        if hasattr(self, 'use_fallback_confidence') and self.use_fallback_confidence:
            logger.warning("Using fallback confidence scoring (softmax-based)")
            # Use softmax confidence as fallback
            embeddings_norm = F.normalize(embeddings, dim=1)
            # Simple distance-based confidence
            distances = torch.cdist(embeddings_norm, embeddings_norm)
            confidence = 1.0 - (distances.mean(dim=1) / distances.max(dim=1)[0].max())
            return torch.clamp(confidence, 0.0, 1.0)
        
        try:
            import hdbscan as hdbscan_lib
        except ImportError:
            # Fallback to simple confidence scoring without hdbscan
            logger.warning("hdbscan not available, using fallback confidence scoring")
            self.use_fallback_confidence = True
            # Use softmax confidence as fallback
            embeddings_norm = F.normalize(embeddings, dim=1)
            # Simple distance-based confidence
            distances = torch.cdist(embeddings_norm, embeddings_norm)
            confidence = 1.0 - (distances.mean(dim=1) / distances.max(dim=1)[0].max())
            return torch.clamp(confidence, 0.0, 1.0)
        from sklearn.decomposition import PCA

        # Phase 1.2: Concatenate raw features with HGNN embeddings to restore per-alert discriminability
        if raw_features is not None:
            raw_np = raw_features.detach().cpu().numpy()
            # Ensure raw_features matches embeddings count
            if raw_np.shape[0] != embeddings.shape[0]:
                raise ValueError(f"raw_features {raw_np.shape[0]} != embeddings {embeddings.shape[0]}")
            # Normalize embeddings first, then concat
            embeddings_np = F.normalize(embeddings, dim=1).detach().cpu().numpy()
            z = np.concatenate([embeddings_np, raw_np], axis=1)
        else:
            z = F.normalize(embeddings, dim=1).detach().cpu().numpy()
            
        # Optional: Soft-ZCA whitening to fix embedding collapse (cosine_sim > 0.95)
        if self.use_zca_whitening:
            # Use embedding-only slice for cosine_sim diagnostic (raw_features inflate dot products)
            emb_only = z[:, :128] if z.shape[1] > 128 else z
            emb_norm = emb_only / (np.linalg.norm(emb_only, axis=1, keepdims=True) + 1e-8)
            pre_sim = float(np.mean(emb_norm @ emb_norm.T) - np.eye(len(emb_norm)).mean())
            z = soft_zca_whiten(z, eps=self.zca_eps)
            emb_only_post = z[:, :128] if z.shape[1] > 128 else z
            emb_norm_post = emb_only_post / (np.linalg.norm(emb_only_post, axis=1, keepdims=True) + 1e-8)
            post_sim = float(np.mean(emb_norm_post @ emb_norm_post.T) - np.eye(len(emb_norm_post)).mean())
            logger.info(
                f"ZCA whitening applied (eps={self.zca_eps}): "
                f"mean_cosine_sim {pre_sim:.4f} → {post_sim:.4f}"
            )
            
        n, d = z.shape

        # Diagnose over-smoothing before proceeding
        if n > 1:
            # Mean pairwise cosine similarity on a sample (expensive for large N)
            sample_size = min(n, 200)
            idx = np.random.choice(n, sample_size, replace=False)
            z_sample = z[idx]
            # Use embedding-only slice for cosine_sim diagnostic (raw_features inflate dot products)
            emb_sample = z_sample[:, :128] if z_sample.shape[1] > 128 else z_sample
            emb_norm_sample = emb_sample / (np.linalg.norm(emb_sample, axis=1, keepdims=True) + 1e-8)
            # Cosine sim matrix (embeddings are L2-normalised, so dot product = cosine sim)
            sim_matrix = emb_norm_sample @ emb_norm_sample.T
            # Upper triangle only (exclude diagonal)
            upper = sim_matrix[np.triu_indices(sample_size, k=1)]
            mean_cosine_sim = float(np.mean(upper))
            if mean_cosine_sim > 0.95:
                logger.warning(
                    f"OVER-SMOOTHING DETECTED: mean pairwise cosine similarity="
                    f"{mean_cosine_sim:.4f} > 0.95. "
                    f"Embeddings have collapsed. Consider reducing num_layers "
                    f"to 1 or adding residual skip connections to MITREHeteroGNN."
                )
                if self.hdbscan_metric_fallback:
                    logger.warning("P3 Fallback triggered: Skipping UMAP/PCA and using direct cosine metric for sparse graph")
                    self.use_umap = False
                    self.pca_components = None
        else:
            mean_cosine_sim = 0.0

        # PCA whitening (amplifies variance in over-smoothed embeddings)

        if self.pca_components is not None and d > self.pca_components and n > self.pca_components:
            self._pca = PCA(
                n_components=self.pca_components,
                whiten=True,   # ← key: normalise variance per component
                random_state=self.seed,
            )
            z_reduced = self._pca.fit_transform(z)
            explained_var = float(self._pca.explained_variance_ratio_.sum())
            logger.info(
                f"PCA: {d}→{self.pca_components} dims, "
                f"explained variance={explained_var:.3f}"
            )
        else:
            z_reduced = z
            logger.info(f"PCA skipped (n={n}, d={d}, pca_components={self.pca_components})")

        # -----------------------------------------------------------------
        # UMAP dimensionality reduction (optional, better topology preservation)
        # -----------------------------------------------------------------
        if self.use_umap:
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=self.umap_n_components,
                    n_neighbors=self.umap_n_neighbors,
                    min_dist=self.umap_min_dist,
                    metric="cosine",
                    random_state=self.seed,
                )
                z_reduced = reducer.fit_transform(z_reduced)
                logger.info(f"UMAP applied: reduced to {self.umap_n_components} dims")
            except ImportError:
                logger.warning("umap-learn not installed. Skipping UMAP reduction.")

        # -----------------------------------------------------------------
        # Clustering
        # -----------------------------------------------------------------
        if self.clustering_method == "spectral" and self.n_clusters is not None:
            from sklearn.cluster import SpectralClustering
            logger.info(f"Using Spectral Clustering with n_clusters={self.n_clusters}")
            sc = SpectralClustering(
                n_clusters=self.n_clusters,
                affinity="cosine",
                random_state=self.seed,
                assign_labels="kmeans",
                n_init=10,
            )
            raw_labels = sc.fit_predict(z_reduced.astype(np.float64))
            # SpectralClustering has no noise points — convert to HDBSCAN-style labels
            clusterer = type('obj', (object,), {'labels_': raw_labels, 'probabilities_': np.ones(len(raw_labels), dtype=np.float32)})()
            self._clusterer = clusterer
            
        elif self.clustering_method == "bgmm":
            from sklearn.mixture import BayesianGaussianMixture
            logger.info(f"Using BayesianGMM with max n_components={self.bgmm_n_components}")
            bgmm = BayesianGaussianMixture(
                n_components=self.bgmm_n_components,
                covariance_type="full",
                random_state=self.seed,
                max_iter=200,
                n_init=3,
            )
            bgmm.fit(z_reduced.astype(np.float64))
            raw_labels = bgmm.predict(z_reduced.astype(np.float64))
            probs = bgmm.predict_proba(z_reduced.astype(np.float64)).max(axis=1).astype(np.float32)
            clusterer = type('obj', (object,), {'labels_': raw_labels, 'probabilities_': probs})()
            self._clusterer = clusterer

        elif self.clustering_method == "prototype" and self.prototype_checkpoint_path:
            from training.train_prototypes import SupervisedPrototypeHead
            logger.info(f"Using Prototype inference from {self.prototype_checkpoint_path}")
            ckpt = torch.load(
                self.prototype_checkpoint_path, map_location="cpu", weights_only=False
            )
            num_classes = ckpt["num_classes"]
            
            # Use original normalized embeddings, not z (which may have raw features appended or be PCA/UMAP reduced)
            emb_norm = F.normalize(embeddings, p=2, dim=1)
            hidden_dim = emb_norm.shape[1]

            proto_head = SupervisedPrototypeHead(
                num_classes=num_classes, hidden_dim=hidden_dim
            )
            proto_head.load_state_dict(ckpt["prototype_state_dict"])
            proto_head.eval()

            with torch.no_grad():
                logits = proto_head(emb_norm)                              # [N, K]
            labels = logits.argmax(dim=1).numpy()
            probabilities = (
                torch.softmax(logits / 0.1, dim=1).max(dim=1).values
                .numpy().astype(np.float32)
            )
            # No noise points for prototype assignments (every sample assigned)
            clusterer = type('obj', (object,), {'labels_': labels, 'probabilities_': probabilities})()
            self._clusterer = clusterer

        else:
            # Default: HDBSCAN
            hdbscan_kwargs = {
                "min_cluster_size": min(self.min_cluster_size, max(2, n // 20)),
                "min_samples": self.min_samples,
                "metric": "cosine",
                "algorithm": "generic",  # boruvka_balltree doesn't support cosine
                "prediction_data": True,
                "cluster_selection_method": self.cluster_selection_method,
                "cluster_selection_epsilon": self.cluster_selection_epsilon,
            }
            # Only add random_state for metrics that support it (not cosine)
            if self.metric != "cosine":
                hdbscan_kwargs["random_state"] = self.seed
            
            clusterer = hdbscan_lib.HDBSCAN(**hdbscan_kwargs)
            # HDBSCAN generic algorithm requires float64
            clusterer.fit(z_reduced.astype(np.float64))
            
            if self.auto_tune and len(set(clusterer.labels_) - {-1}) <= 1:
                candidates = [c for c in [30, 20, 15, 10, 5] if c < self.min_cluster_size]
                for candidate_mcs in candidates:
                    retry_kwargs = {
                        "min_cluster_size": candidate_mcs,
                        "min_samples": max(1, candidate_mcs // 3),
                        "metric": "cosine",
                        "algorithm": "generic",
                        "prediction_data": True,
                        "cluster_selection_method": self.cluster_selection_method,
                    }
                    # Only add random_state for metrics that support it (not cosine)
                    if self.metric != "cosine":
                        retry_kwargs["random_state"] = self.seed
                    
                    retry_clusterer = hdbscan_lib.HDBSCAN(**retry_kwargs)
                    retry_clusterer.fit(z_reduced.astype(np.float64))
                    n_found = len(set(retry_clusterer.labels_) - {-1})
                    logger.info(f"auto_tune: min_cluster_size={candidate_mcs} → {n_found} clusters")
                    if n_found >= 2:
                        clusterer = retry_clusterer
                        break
                if len(set(clusterer.labels_) - {-1}) <= 1:
                    logger.warning("auto_tune: could not find >1 cluster at any min_cluster_size")
    
            self._clusterer = clusterer

        n_found = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
        n_noise = int((clusterer.labels_ == -1).sum())
        logger.info(
            f"HDBSCAN: found {n_found} clusters, "
            f"{n_noise}/{n} noise points ({n_noise/n:.1%})"
        )

        # -----------------------------------------------------------------
        # v2.1: Use all_points_membership_vectors() for full probability matrix.
        # Returns [N, n_clusters] where every point (noise, border, core) gets
        # a real probability distribution. No hard 0.0 values, no noise mask needed.
        try:
            import hdbscan as hdbscan_lib
            membership_vectors = hdbscan_lib.all_points_membership_vectors(clusterer)
            # membership_vectors shape: [N, n_clusters]
            # confidence = max probability across all clusters for each point
            confidence = membership_vectors.max(axis=1).astype(np.float32)
            
            # FIX: Replace any NaN values with clusterer probabilities
            if np.isnan(confidence).any():
                nan_mask = np.isnan(confidence)
                confidence[nan_mask] = clusterer.probabilities_[nan_mask]
                logger.warning(f"Fixed {nan_mask.sum()} NaN confidence values")
            
            logger.info(
                f"all_points_membership_vectors: shape={membership_vectors.shape}, "
                f"conf mean={confidence.mean():.3f}, min={confidence.min():.3f}, "
                f"max={confidence.max():.3f}"
            )
        except Exception as exc:
            logger.warning(
                f"all_points_membership_vectors() failed ({exc}). "
                f"Falling back to clusterer.probabilities_."
            )
            confidence = clusterer.probabilities_.astype(np.float32)
        
        # Final safety: ensure no NaN remains
        if np.isnan(confidence).any():
            logger.error(f"Confidence still contains {np.isnan(confidence).sum()} NaN values - using uniform fallback")
            confidence = np.full(n, 0.5, dtype=np.float32)

        # Fallback: if all noise or single cluster, return moderate uniform score
        if n_found <= 1 and self.fallback_to_uniform:
            logger.warning(
                f"HDBSCAN found {n_found} cluster(s) — returning uniform confidence=0.5."
            )
            confidence = np.full(n, 0.5, dtype=np.float32)

        return confidence

    def score(self, embeddings: torch.Tensor) -> np.ndarray:
        """Alias for fit_score (HDBSCAN always fits and scores together)."""
        return self.fit_score(embeddings)


# ============================================================================
# HGNNCorrelationEngine — main engine with confidence-gated UF fallback (v2.1)
# ============================================================================

class HGNNCorrelationEngine:
    """
    Drop-in replacement for Union-Find correlation engine.

    Primary path: HGNN inference → cluster_logits → cluster_confidence →
                  pred_cluster assigned via argmax.

    Refinement path (new in v2.1): alerts with cluster_confidence below
    ``confidence_gate`` are re-correlated by enhanced_correlation() using
    confidence_guided_threshold() to derive the UF threshold.  The
    motivating result is the sensitivity study finding that threshold ≥ 0.7
    pushed ARI to 0.9708 — the HGNN's own confidence is the most reliable
    signal for when to apply a tighter threshold.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temperature: float = 1.0,
        confidence_gate: float = _DEFAULT_CONFIDENCE_GATE,
        uf_usernames: Optional[List[str]] = None,
        uf_addresses: Optional[List[str]] = None,
        use_geometric_confidence: bool = True,
        hdbscan_min_cluster_size: int = 5,
        hdbscan_pca_components: int = 32,
        use_uf_refinement: bool = False,
        pure_unsupervised: bool = True,
        noise_point_strategy: str = "zero",
        hdbscan_auto_tune: bool = False,
        hdbscan_cluster_selection_epsilon: float = 0.0,
        hdbscan_use_umap: bool = False,
        hdbscan_umap_n_components: int = 10,
        hdbscan_umap_n_neighbors: int = 15,
        hdbscan_umap_min_dist: float = 0.1,
        hdbscan_cluster_selection_method: str = "eom",
        hdbscan_metric_fallback: bool = False,
        use_temporal_memory: bool = False,
        seed: int = 42,
        build_bridge_edges: bool = True,
        collapse_entities: bool = False,
        hdbscan_zca_whitening: bool = False,
        hdbscan_zca_eps: float = 0.1,
        clustering_method: str = "hdbscan",
        hdbscan_n_clusters: Optional[int] = None,
        bgmm_n_components: int = 30,
        prototype_checkpoint_path: Optional[str] = None,
        aggr_method: str = "mean",
        use_burstiness: bool = False,
        track_data_source: bool = False,
        build_precedes_edges: bool = False,
        precedes_window_hours: float = 2.0,
    ):
        """
        Args
        ----
        model_path : Optional[str]
            Path to a pretrained HGNN checkpoint.
        hidden_dim : int
            Hidden dimension for MITREHeteroGNN.
        num_heads : int
            Number of attention heads per GATConv layer.
        num_layers : int
            Number of layers for MITREHeteroGNN (default 1 to prevent over-smoothing).
        device : str
            Torch device string ('cuda' or 'cpu').
        temperature : float
            Initial temperature scaling value (refined via calibrate_temperature).
        confidence_gate : float
            Alerts with max-softmax confidence below this value are passed through
            the UF refinement pass.  Default: 0.6.
        uf_usernames : Optional[List[str]]
            Username columns to use in the UF refinement pass.
            Defaults to ["SourceHostName", "DeviceHostName", "DestinationHostName"].
        uf_addresses : Optional[List[str]]
            Address columns to use in the UF refinement pass.
            Defaults to ["SourceAddress", "DestinationAddress", "DeviceAddress"].
        use_geometric_confidence : bool
            Use Geometry-Aware Embedding Confidence (GAEC) instead of max-softmax.
        hdbscan_min_cluster_size : int
            Minimum cluster size for HDBSCAN.
        hdbscan_pca_components : int
            Number of PCA components for HDBSCAN preprocessing.
        use_uf_refinement : bool
            When False (default), all alerts keep their HGNN cluster assignment
            regardless of confidence. This is the empirically validated default
            for the UNSW-NB15 checkpoint: ARI=0.4042 vs 0.3541 with UF enabled,
            and singleton_fraction=1.0 when UF is active (v2.6, confirmed v2.9).
            Set True only to explicitly test the hybrid UF path or when using a
            checkpoint with genuinely dispersed embeddings (p25_confidence < 0.1).
        noise_point_strategy : str
            How to handle HDBSCAN noise points (probability=0.0) in the GAEC scorer.
            "zero"        — keep confidence=0.0 (v2.5 behavior, routes to UF).
            "soft_assign" — assign nearest-neighbor cosine confidence in [0.05, 0.4].
            Only relevant when use_geometric_confidence=True.
        """
        self.device = device
        if not pure_unsupervised:
            raise ValueError("MITRE-CORE V3 requires pure_unsupervised=True for inference.")
        if clustering_method == "prototype":
            raise ValueError("Prototype inference is not permitted in MITRE-CORE V3.")
        
        self.use_temporal_memory = use_temporal_memory
        if use_temporal_memory:
            from .temporal_enrichment import EntityMemoryModule
            self.entity_memory = EntityMemoryModule().to(device)
        else:
            self.entity_memory = None
            
        self.converter = AlertToGraphConverter(
            entity_memory=self.entity_memory,
            build_bridge_edges=build_bridge_edges,
            collapse_entities=collapse_entities,
            use_burstiness=use_burstiness,
            track_data_source=track_data_source,
            build_precedes_edges=build_precedes_edges,
            precedes_window_hours=precedes_window_hours,
        )
        
        self.temperature = temperature
        self.confidence_gate = confidence_gate
        self.use_geometric_confidence = use_geometric_confidence
        self.use_uf_refinement = use_uf_refinement
        self.pure_unsupervised = pure_unsupervised
        self.prototype_checkpoint_path = prototype_checkpoint_path
        self.seed = seed
        self.confidence_scorer = EmbeddingConfidenceScorer(
            min_cluster_size=hdbscan_min_cluster_size,
            pca_components=hdbscan_pca_components,
            min_samples=3,
            metric="cosine",
            fallback_to_uniform=True,
            noise_point_strategy=noise_point_strategy,
            auto_tune=hdbscan_auto_tune,
            cluster_selection_epsilon=hdbscan_cluster_selection_epsilon,
            use_umap=hdbscan_use_umap,
            umap_n_components=hdbscan_umap_n_components,
            umap_n_neighbors=hdbscan_umap_n_neighbors,
            umap_min_dist=hdbscan_umap_min_dist,
            cluster_selection_method=hdbscan_cluster_selection_method,
            use_zca_whitening=hdbscan_zca_whitening,
            zca_eps=hdbscan_zca_eps,
            clustering_method=clustering_method,
            n_clusters=hdbscan_n_clusters,
            bgmm_n_components=bgmm_n_components,
            prototype_checkpoint_path=self.prototype_checkpoint_path,
            seed=self.seed,
        ) if use_geometric_confidence else None

        # Default UF column lists (mirror correlation_indexer.py main())
        self.uf_usernames = uf_usernames or [
            "SourceHostName", "DeviceHostName", "DestinationHostName"
        ]
        self.uf_addresses = uf_addresses or [
            "SourceAddress", "DestinationAddress", "DeviceAddress"
        ]

        # Initialize model (will be recreated if checkpoint has different config)
        self.model = MITREHeteroGNN(
            alert_feature_dim=6,  # Default: 6 base features
            hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
            aggr_method=aggr_method
        ).to(device)

        if model_path:
            # For prototype mode: load backbone from prototype checkpoint, not from model_path
            if clustering_method == "prototype" and prototype_checkpoint_path and \
                    Path(prototype_checkpoint_path).exists():
                logger.info(
                    f"Prototype mode: loading HGNN backbone from prototype checkpoint "
                    f"{prototype_checkpoint_path} (ignoring model_path={model_path})"
                )
                ckpt = torch.load(prototype_checkpoint_path, map_location=device, weights_only=False)
                state_key = 'hgnn_state_dict'   # key saved by train_prototypes.py
                # Use the prototype checkpoint for model loading
                model_path_to_load = prototype_checkpoint_path
            else:
                ckpt = torch.load(model_path, map_location=device, weights_only=False)
                state_key = 'model_state_dict'
                model_path_to_load = model_path

            # Let the checkpoint dictate the number of clusters and domain heads if available
            vocab_sizes = None
            try:
                state_dict = ckpt.get(state_key, ckpt)

                # Load vocab_sizes if available (for CategoricalAlertEncoder)
                vocab_sizes = ckpt.get('vocab_sizes', None)
                if vocab_sizes:
                    logger.info(f"Loaded vocab_sizes from checkpoint: {vocab_sizes}")

                # Detect alert_feature_dim from checkpoint if available
                ckpt_alert_feature_dim = None
                alert_raw_weight_key = "alert_raw_proj.weight"
                if alert_raw_weight_key in state_dict:
                    ckpt_alert_feature_dim = state_dict[alert_raw_weight_key].shape[1]
                    logger.info(f"Detected alert_feature_dim={ckpt_alert_feature_dim} from checkpoint.")

                # Check for cluster_classifier shape mismatch
                k_weight = "cluster_classifier.3.weight"
                ckpt_clusters = None
                needs_reinit = False
                if k_weight in state_dict:
                    ckpt_clusters = state_dict[k_weight].shape[0]
                    model_clusters = self.model.cluster_classifier[3].weight.shape[0]
                    if ckpt_clusters != model_clusters:
                        logger.info(f"Re-initializing model with {ckpt_clusters} clusters to match checkpoint.")
                        needs_reinit = True
                    elif vocab_sizes:
                        logger.info("Re-initializing model with CategoricalAlertEncoder from checkpoint.")
                        needs_reinit = True

                # Also reinit if alert_feature_dim differs
                if ckpt_alert_feature_dim is not None and ckpt_alert_feature_dim != self.model.alert_feature_dim:
                    logger.info(f"Re-initializing model with alert_feature_dim={ckpt_alert_feature_dim} to match checkpoint.")
                    needs_reinit = True

                if needs_reinit:
                    self.model = MITREHeteroGNN(
                        alert_feature_dim=ckpt_alert_feature_dim if ckpt_alert_feature_dim is not None else 6,
                        hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
                        num_clusters=ckpt_clusters if ckpt_clusters is not None else self.model.num_clusters,
                        vocab_sizes=vocab_sizes,
                        aggr_method=aggr_method
                    ).to(device)
            except Exception as e:
                logger.warning(f"Failed to inspect checkpoint for num_clusters/vocab_sizes: {e}")

            self._load_checkpoint(model_path_to_load, state_key)

        self.model.eval()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_checkpoint(self, model_path: str, state_key: str = "model_state_dict") -> None:
        try:
            # Handle UninitializedParameter in checkpoints
            import torch.nn as nn
            
            # Add safe globals for UninitializedParameter
            torch.serialization.add_safe_globals([nn.UninitializedParameter])
            
            state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if state_key in state_dict:
                state_dict = state_dict[state_key]
                
            model_dict = self.model.state_dict()
            filtered = {}
            for k, v in state_dict.items():
                if k in model_dict:
                    # Skip uninitialized parameters in the checkpoint itself
                    if isinstance(v, nn.UninitializedParameter):
                        continue
                    
                    param = model_dict[k]
                    if isinstance(param, nn.UninitializedParameter) or param.shape == v.shape:
                        filtered[k] = v
                    else:
                        logger.warning(f"Shape mismatch for {k}: model={param.shape}, checkpoint={v.shape}")
            
            # Load with strict=False to allow missing/extra keys (e.g., new layer_norms)
            incompatible = self.model.load_state_dict(filtered, strict=False)
            if incompatible.missing_keys:
                logger.info(f"Missing keys (will use random init): {list(incompatible.missing_keys)[:5]}...")
            if incompatible.unexpected_keys:
                logger.info(f"Unexpected keys (skipped): {list(incompatible.unexpected_keys)[:5]}...")
            
            logger.info(
                f"Loaded checkpoint {model_path} "
                f"({len(filtered)}/{len(state_dict)} keys loaded, "
                f"{len(incompatible.missing_keys)} missing, "
                f"{len(incompatible.unexpected_keys)} unexpected)"
            )
        except Exception as exc:
            logger.warning(f"Could not fully load checkpoint: {exc}")

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / max(self.temperature, 1e-6)

    def _log_confidence_diagnostics(
        self,
        confidence_scores: np.ndarray,
        source: str,
        dataset_name: str = "unknown",
    ) -> None:
        """
        Log confidence distribution statistics to help diagnose gate behavior.
        Writes to both the Python logger and a diagnostics CSV for persistence.
        """
        import json
        from pathlib import Path

        p25 = float(np.percentile(confidence_scores, 25))
        p75 = float(np.percentile(confidence_scores, 75))
        mean = float(np.mean(confidence_scores))
        std = float(np.std(confidence_scores))
        gate = self.confidence_gate

        # Mirror the exact formula used in correlation_indexer.confidence_guided_threshold()
        # so diagnostic predictions match runtime behaviour.
        adjustment = mean - 0.5
        derived_threshold = float(np.clip(0.3 + adjustment, 0.1, 0.9))

        logger.info(f"\n--- Confidence Diagnostics [{source}] ---")
        logger.info(f"  dataset    : {dataset_name}")
        logger.info(f"  mean       : {mean:.4f}")
        logger.info(f"  std        : {std:.4f}")
        logger.info(f"  p25        : {p25:.4f}")
        logger.info(f"  p75        : {p75:.4f}")
        logger.info(f"  gate       : {gate:.4f}")
        logger.info(f"  pct < gate : {(confidence_scores < gate).mean():.2%}")
        logger.info(f"  → UF threshold will be: {derived_threshold:.4f}")

        if std < 0.01:
            logger.warning(
                f"  ⚠ NEAR-ZERO VARIANCE (std={std:.4f}). "
                f"Gate sweep will be flat. "
                f"Check: (1) GAEC enabled? (2) k-means converging? "
                f"(3) Embeddings collapsing (over-smoothing)?"
            )
        if mean < 0.2:
            logger.warning(
                f"  ⚠ VERY LOW MEAN CONFIDENCE ({mean:.4f}). "
                f"If source=softmax, this means classification head is untrained. "
                f"If source=gaec, this means embeddings are highly dispersed — "
                f"increase n_centroids or check for over-smoothing."
            )

        # Persist to diagnostics log
        log_path = Path("experiments/results/confidence_diagnostics.jsonl")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "dataset": dataset_name,
                "source": source,
                "mean": mean,
                "std": std,
                "p25": p25,
                "p75": p75,
                "gate": gate,
                "pct_below_gate": float((confidence_scores < gate).mean()),
                "derived_uf_threshold": derived_threshold,
                "n_hdbscan_clusters": int(getattr(
                    getattr(self, 'confidence_scorer', None),
                    '_clusterer', None
                ).labels_.max() + 1) if (
                    self.confidence_scorer is not None and
                    self.confidence_scorer._clusterer is not None
                ) else -1,
            }) + "\n")

    def _uf_refinement_pass(
        self,
        df: pd.DataFrame,
        confidence: np.ndarray,
        cluster_offset: int,
        full_confidence: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Run enhanced_correlation() on a subset of low-confidence alerts,
        using confidence_guided_threshold() to determine the UF threshold.

        This is the bridge between the HGNN output and the Union-Find engine
        introduced in v2.1.  It is intentionally isolated so it can be unit-
        tested independently of the full engine.

        Args
        ----
        df : pd.DataFrame
            Subset of the original alert DataFrame — only low-confidence rows.
        confidence : np.ndarray
            Per-alert confidence values for this subset, shape [M,].
        cluster_offset : int
            Integer offset added to UF cluster IDs to avoid collisions with
            the HGNN cluster IDs already assigned to high-confidence alerts.
        full_confidence : Optional[np.ndarray]
            Full per-alert confidence array for threshold computation.
            If provided, used instead of `confidence` for confidence_guided_threshold().
            This ensures the threshold is computed from the full distribution
            rather than just the low-confidence subset.

        Returns
        -------
        pd.DataFrame
            ``df`` with ``pred_cluster`` overwritten by the UF result
            (offset-adjusted) and ``cluster_confidence`` preserved.
        """
        from core.correlation_indexer import enhanced_correlation

        # Determine which UF columns are actually present in this subset
        present_usernames = [c for c in self.uf_usernames if c in df.columns]
        present_addresses = [c for c in self.uf_addresses if c in df.columns]

        if not present_addresses and not present_usernames:
            logger.warning(
                "UF refinement pass skipped: no address or username columns found in subset."
            )
            return df

        # Use full confidence array for threshold computation if available
        # This fixes the bug where only low-confidence subset was used,
        # causing threshold to always hit floor (0.1)
        threshold_confidence = full_confidence if full_confidence is not None else confidence

        uf_result = enhanced_correlation(
            data=df.reset_index(drop=True),
            usernames=present_usernames,
            addresses=present_addresses,
            use_temporal="EndDate" in df.columns,
            use_adaptive_threshold=False,   # confidence_guided_threshold takes over
            threshold_override=None,
            cluster_confidence=threshold_confidence,
        )

        # Offset UF cluster IDs to avoid collision with HGNN cluster space
        uf_result["pred_cluster"] = uf_result["pred_cluster"] + cluster_offset

        # Re-attach confidence scores (UF doesn't produce them — preserve HGNN's)
        uf_result["cluster_confidence"] = confidence

        logger.info(
            f"UF refinement pass: {len(df)} low-confidence alerts → "
            f"{uf_result['pred_cluster'].nunique()} sub-clusters "
            f"(threshold_source={uf_result['threshold_source'].iloc[0]}, "
            f"threshold={uf_result['correlation_threshold_used'].iloc[0]:.4f})"
        )

        return uf_result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def calibrate_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> float:
        """
        Learn optimal temperature via NLL minimisation (Guo et al., ICML 2017).

        Args
        ----
        logits : torch.Tensor [N, C]
            Raw model logits.
        labels : torch.Tensor [N]
            Ground-truth cluster indices.
        lr : float
            Learning rate for LBFGS optimiser.
        max_iter : int
            Maximum optimisation iterations.

        Returns
        -------
        float
            Optimal temperature value (also stored as self.temperature).
        """
        temperature = nn.Parameter(torch.ones(1, device=self.device))
        optimiser = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

        def eval_nll():
            optimiser.zero_grad()
            loss = F.cross_entropy(logits / temperature.clamp(min=1e-6), labels)
            loss.backward()
            return loss

        optimiser.step(eval_nll)
        self.temperature = float(temperature.item())
        logger.info(f"Temperature calibration complete: T={self.temperature:.4f}")
        return self.temperature

    def extract_embeddings(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract HGNN backbone embeddings for a dataframe of alerts.
        Useful for fair baseline comparisons.
        """
        self.model.eval()
        graph_data = self.converter.convert(df)
        graph_data = graph_data.to(self.device)
        
        # Dynamically pad or truncate node features
        for ntype in graph_data.node_types:
            encoder_name = f"{ntype}_encoder"
            if hasattr(self.model, encoder_name):
                encoder = getattr(self.model, encoder_name)
                if hasattr(encoder, "in_channels") and encoder.in_channels > 0:
                    expected_dim = encoder.in_channels
                    current_dim = graph_data[ntype].x.shape[1]
                    if current_dim < expected_dim:
                        graph_data[ntype].x = torch.nn.functional.pad(graph_data[ntype].x, (0, expected_dim - current_dim))
                    elif current_dim > expected_dim:
                        graph_data[ntype].x = graph_data[ntype].x[:, :expected_dim]
                        
        with torch.no_grad():
            _, x_dict = self.model(graph_data)
            alert_embeddings = x_dict["alert"].cpu().numpy()
            
        return alert_embeddings

    def correlate(self, df: pd.DataFrame, explain: bool = False, embed_only: bool = False) -> pd.DataFrame:
        """
        Correlate alerts using the HGNN with a confidence-gated UF refinement pass.

        Pipeline
        --------
        1. Convert ``df`` to HeteroData via AlertToGraphConverter.
        2. Run MITREHeteroGNN forward pass → cluster_logits.
        3. Apply temperature scaling → cluster_probs → cluster_confidence.
        4. Assign pred_cluster via argmax for all alerts.
        5. Identify low-confidence alerts (confidence < self.confidence_gate).
        6. For low-confidence alerts: run _uf_refinement_pass() which calls
           enhanced_correlation() with confidence_guided_threshold() driving
           the UF threshold.
        7. Merge high-confidence HGNN assignments with UF-refined assignments.

        The motivation for step 6 is the MITRE-CORE v2 sensitivity result:
        threshold ≥ 0.7 → ARI 0.9708, vs. HGNN Full ARI 0.6174.  The HGNN's
        own confidence is the best available signal for when its embedding space
        is unreliable — passing those alerts to a threshold-aware UF pass closes
        the gap without regressing on high-confidence predictions.

        Args
        ----
        df : pd.DataFrame
            Alert DataFrame. Must contain at least: AlertId, MalwareIntelAttackType,
            AttackSeverity, EndDate.  Address and hostname columns are used if present.
        explain : bool
            If True, returns additional explainability information including
            attention weights and feature importance for cluster assignments.
        embed_only : bool
            If True, only returns alert embeddings without clustering. Used for
            two-phase processing where embeddings are collected from chunks first,
            then clustered together to avoid HDBSCAN fragmentation.

        Returns
        -------
        pd.DataFrame or tuple
            If embed_only=False: Copy of ``df`` with added columns:
              - pred_cluster          : integer cluster ID.
              - cluster_confidence    : HGNN max-softmax confidence [0, 1].
              - correlation_method    : 'hgnn' | 'hgnn+uf_refinement'.
            If embed_only=True: Tuple of (embeddings numpy array, alert_ids list).
        """
        logger.info(f"Building heterogeneous graph from {len(df)} alerts...")
        graph_data = self.converter.convert(df)
        graph_data = graph_data.to(self.device)

        logger.info(f"Graph: {graph_data['alert'].num_nodes} alerts, {len(graph_data.edge_types)} edge types")
        
        # Capture raw alert features for Phase 1.2 concat before HGNN forward pass
        raw_alert_features = graph_data["alert"].x.clone()
        
        # Dynamically pad or truncate node features to match the loaded model's encoder expectations
        for ntype in graph_data.node_types:
            encoder_name = f"{ntype}_encoder"
            if hasattr(self.model, encoder_name):
                encoder = getattr(self.model, encoder_name)
                # Check if encoder has been initialized
                if hasattr(encoder, "in_channels") and encoder.in_channels > 0:
                    expected_dim = encoder.in_channels
                    current_dim = graph_data[ntype].x.shape[1]
                    if current_dim < expected_dim:
                        graph_data[ntype].x = torch.nn.functional.pad(graph_data[ntype].x, (0, expected_dim - current_dim))
                    elif current_dim > expected_dim:
                        graph_data[ntype].x = graph_data[ntype].x[:, :expected_dim]

        with torch.no_grad():
            # ------------------------------------------------------------------
            # Step 1-4: HGNN inference
            # ------------------------------------------------------------------
            self.model.eval()
            
            cluster_logits, x_dict = self.model(graph_data)
            alert_embeddings = x_dict["alert"]  # [N, hidden_dim] from forward()

            # If embed_only mode, return embeddings immediately without clustering
            if embed_only:
                alert_ids = df["AlertId"].tolist() if "AlertId" in df.columns else list(range(len(df)))
                return alert_embeddings.cpu().numpy(), alert_ids

            if self.use_geometric_confidence and self.confidence_scorer is not None:
                # Use geometry-aware embedding confidence (GAEC) instead of softmax.
                # alert_embeddings come from the message-passing layers directly,
                # before the classification head - no calibration required.
                confidence_scores = self.confidence_scorer.fit_score(
                    alert_embeddings,
                    confidence_gate=self.confidence_gate,
                    raw_features=raw_alert_features,  # Phase 1.2: concat raw features before PCA/UMAP
                )
                confidence_source = "gaec"
                # Convert torch.Tensor to numpy if needed
                if hasattr(confidence_scores, 'cpu'):
                    confidence_scores = confidence_scores.cpu().numpy()

                if self.pure_unsupervised:
                    logger.info("pure_unsupervised=True: using HDBSCAN labels")
                    if self.confidence_scorer._clusterer is not None:
                        hdbscan_labels = self.confidence_scorer._clusterer.labels_
                        cluster_preds = hdbscan_labels.copy()

                        noise_mask = cluster_preds == -1
                        if noise_mask.any():
                            if self.use_uf_refinement:
                                # UF logic will handle noise points below
                                pass
                            else:
                                if len(set(cluster_preds[~noise_mask])) > 0:
                                    cluster_preds[noise_mask] = 0
                                else:
                                    cluster_preds[noise_mask] = 0
                        cluster_preds = torch.tensor(cluster_preds, device=self.device)
                    else:
                        logger.warning("HDBSCAN _clusterer is None. Falling back to cluster 0.")
                        cluster_preds = torch.zeros(len(df), dtype=torch.long, device=self.device)
            else:
                # Softmax mode: use the HGNN classification head directly.
                # Required for supervised checkpoints (multidomain_v2) where the
                # 10-cluster head encodes campaign structure — GAEC/HDBSCAN cannot
                # recover this on sparse graphs (e.g. UNSW-NB15 with 175K unique IPs).
                probs = torch.softmax(cluster_logits, dim=-1)          # [N, n_clusters]
                cluster_preds = torch.argmax(probs, dim=-1)            # [N]
                confidence_scores = probs.max(dim=-1).values.cpu().numpy()  # [N]
                confidence_source = "softmax"
                logger.info(f"Softmax mode: {cluster_preds.unique().numel()} HGNN clusters, "
                            f"avg confidence={confidence_scores.mean():.3f}")

        self._log_confidence_diagnostics(
            confidence_scores,
            source=confidence_source,
            dataset_name=getattr(df, "_dataset_name", "unknown"),
        )

        result_df = df.copy()
        result_df["pred_cluster"] = cluster_preds.cpu().numpy()
        result_df["cluster_confidence"] = confidence_scores
        result_df["confidence_source"] = confidence_source
        result_df["correlation_method"] = "hgnn"

        avg_conf = float(np.mean(confidence_scores))
        n_clusters_hgnn = int(cluster_preds.unique().numel())
        logger.info(
            f"HGNN: {n_clusters_hgnn} clusters, "
            f"avg confidence={avg_conf:.3f}, "
            f"T={self.temperature:.3f}"
        )

        # ------------------------------------------------------------------
        # Step 5–7: Confidence-gated UF refinement pass
        # ------------------------------------------------------------------
        low_conf_mask = confidence_scores < self.confidence_gate
        n_low_conf = int(low_conf_mask.sum())

        if not self.use_uf_refinement:
            # v2.6: UF disabled — all alerts retain their HGNN cluster assignment.
            # Noise points (confidence=0.0) keep argmax prediction rather than
            # becoming singletons in the UF pass. This is the H-A test.
            logger.info(
                f"UF refinement DISABLED (use_uf_refinement=False). "
                f"All {len(df)} alerts retain HGNN assignments. "
                f"({n_low_conf} would have been routed to UF at gate={self.confidence_gate})"
            )
        elif n_low_conf > 0:
            logger.info(
                f"UF refinement pass triggered: {n_low_conf}/{len(df)} alerts "
                f"below confidence_gate={self.confidence_gate}"
            )

            low_conf_df = df[low_conf_mask].copy()
            low_conf_confidence = confidence_scores[low_conf_mask]

            # Cluster offset = max HGNN cluster ID + 1 to avoid ID collision
            cluster_offset = int(result_df["pred_cluster"].max()) + 1

            uf_refined = self._uf_refinement_pass(
                df=low_conf_df,
                confidence=low_conf_confidence,
                cluster_offset=cluster_offset,
                full_confidence=confidence_scores,
            )

            # Write UF-refined assignments back into result_df
            result_df.loc[low_conf_mask, "pred_cluster"] = uf_refined["pred_cluster"].values
            result_df.loc[low_conf_mask, "correlation_method"] = "hgnn+uf_refinement"
            
            if "correlation_threshold_used" in uf_refined.columns:
                if "correlation_threshold_used" not in result_df.columns:
                    result_df["correlation_threshold_used"] = float("nan")
                result_df.loc[low_conf_mask, "correlation_threshold_used"] = uf_refined["correlation_threshold_used"].values
                
            if "threshold_source" in uf_refined.columns:
                if "threshold_source" not in result_df.columns:
                    result_df["threshold_source"] = float("nan")
                result_df.loc[low_conf_mask, "threshold_source"] = uf_refined["threshold_source"].values

            n_clusters_final = result_df["pred_cluster"].nunique()
            logger.info(
                f"After UF refinement: {n_clusters_final} total clusters "
                f"({n_clusters_hgnn} from HGNN, "
                f"{uf_refined['pred_cluster'].nunique()} from UF refinement)"
            )
        else:
            logger.info(
                f"All {len(df)} alerts above confidence_gate={self.confidence_gate} "
                f"— UF refinement pass skipped."
            )

        # ------------------------------------------------------------------
        # Step 8: Explainability extraction (if requested)
        # ------------------------------------------------------------------
        if explain:
            logger.info("Extracting explainability information...")
            try:
                from .hgnn_explainability import AttentionExtractor
                attention_extractor = AttentionExtractor()
                attention_extractor.register_hooks(self.model)
                
                # Re-run forward pass to capture attention
                with torch.no_grad():
                    self.model.eval()
                    cluster_logits, x_dict = self.model(graph_data, domain=domain)
                
                attention_weights = attention_extractor.get_attention_weights()
                attention_extractor.clear_hooks()
                
                # Add attention information to result
                result_df["attention_info"] = None  # Placeholder for per-alert attention
                
                logger.info(f"Captured attention weights from {len(attention_weights)} layers")
                
            except ImportError:
                logger.warning("Explainability module not available, skipping attention extraction")
            except Exception as e:
                logger.error(f"Attention extraction failed: {e}")

        return result_df

    def get_attention_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyse which edge types contributed most to clustering (interpretability)."""
        data = self.converter.convert(df).to(self.device)
        attention_weights = self.model.get_attention_weights(data)
        return {
            str(et): {
                "mean_attention": float(attn.mean()),
                "max_attention": float(attn.max()),
                "attention_std": float(attn.std()),
            }
            for et, attn in attention_weights.items()
        }

    def save_model(self, path: str) -> None:
        """Save trained model weights."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Saved HGNN model to {path}")

    def cluster_embeddings(self, all_embeddings: np.ndarray, all_ids: list, 
                          confidence_gate: float = 0.5) -> pd.DataFrame:
        """
        Cluster pre-collected embeddings using GAEC/HDBSCAN or softmax.
        
        This method is used for two-phase processing to avoid HDBSCAN fragmentation
        when processing chunks. Embeddings are collected first, then clustered together.
        
        Args:
            all_embeddings: numpy array of shape [N, hidden_dim] containing all embeddings
            all_ids: list of alert IDs corresponding to embeddings
            confidence_gate: confidence threshold for GAEC gating
            
        Returns:
            pd.DataFrame with columns: AlertId, pred_cluster, cluster_confidence, correlation_method
        """
        logger.info(f"Clustering {len(all_embeddings)} collected embeddings...")
        
        # Convert embeddings to tensor for confidence scorer
        if isinstance(all_embeddings, np.ndarray):
            all_embeddings_tensor = torch.from_numpy(all_embeddings).float()
        else:
            all_embeddings_tensor = all_embeddings
            
        # Move to device if needed
        if hasattr(self, 'device'):
            all_embeddings_tensor = all_embeddings_tensor.to(self.device)
        
        with torch.no_grad():
            if self.use_geometric_confidence and self.confidence_scorer is not None:
                # Use GAEC/HDBSCAN on collected embeddings
                logger.info("Using GAEC/HDBSCAN on collected embeddings")
                
                # For GAEC, we need raw features - but we don't have them in embed_only mode
                # Since raw features are concatenated with embeddings in fit_score,
                # they need to have shape [N, num_raw_features], but the code expects 
                # num_raw_features to match whatever was used during training or just work with None.
                # Setting it to None bypasses the raw features concatenation step in fit_score.
                raw_features = None
                
                confidence_scores = self.confidence_scorer.fit_score(
                    all_embeddings_tensor,
                    confidence_gate=confidence_gate,
                    raw_features=raw_features,
                )
                confidence_source = "gaec"
                
                if hasattr(confidence_scores, 'cpu'):
                    confidence_scores = confidence_scores.cpu().numpy()
                
                if self.pure_unsupervised:
                    logger.info("pure_unsupervised=True: using HDBSCAN labels")
                    if self.confidence_scorer._clusterer is not None:
                        hdbscan_labels = self.confidence_scorer._clusterer.labels_
                        cluster_preds = hdbscan_labels.copy()
                        
                        noise_mask = cluster_preds == -1
                        if noise_mask.any():
                            if len(set(cluster_preds[~noise_mask])) > 0:
                                cluster_preds[noise_mask] = 0
                            else:
                                cluster_preds[noise_mask] = 0
                    else:
                        logger.warning("HDBSCAN _clusterer is None. Falling back to cluster 0.")
                        cluster_preds = np.zeros(len(all_embeddings), dtype=int)
                else:
                    # For GAEC mode, use confidence scores to assign clusters
                    # This is a simplified approach - in practice, you might need more sophisticated logic
                    cluster_preds = np.zeros(len(all_embeddings), dtype=int)
                    logger.info("GAEC mode: assigning all to cluster 0 (simplified)")
                    
            else:
                # Softmax mode - we don't have cluster logits, so use k-means as fallback
                logger.warning("Softmax mode not available for collected embeddings, using k-means fallback")
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=10, random_state=42)
                cluster_preds = kmeans.fit_predict(all_embeddings)
                confidence_scores = np.ones(len(all_embeddings)) * 0.8  # Placeholder confidence
                confidence_source = "kmeans_fallback"
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'AlertId': all_ids,
            'pred_cluster': cluster_preds,
            'cluster_confidence': confidence_scores,
            'correlation_method': f'hgnn_{confidence_source}'
        })
        
        logger.info(f"Clustering complete: {len(set(cluster_preds))} clusters, "
                   f"avg confidence={np.mean(confidence_scores):.3f}")
        
        return result_df


# ============================================================================
# Training Components (self-supervised pre-training)
# ============================================================================

class ContrastiveAlertLearner(nn.Module):
    """
    Self-supervised contrastive learning for alert embeddings.
    Based on CARLA: Self-Supervised Contrastive Representation Learning (2023-2024).
    """

    def __init__(self, hgnn: MITREHeteroGNN, temperature: float = 0.5):
        super().__init__()
        self.hgnn = hgnn
        self.temperature = temperature

    def forward(self, data1: HeteroData, data2: HeteroData) -> torch.Tensor:
        _, emb1 = self.hgnn(data1)
        _, emb2 = self.hgnn(data2)
        z1 = F.normalize(emb1["alert"], dim=1)
        z2 = F.normalize(emb2["alert"], dim=1)
        sim = torch.mm(z1, z2.t()) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return F.cross_entropy(sim, labels)


class GraphAugmenter:
    """
    Data augmentation module for graph-based alert data.
    
    NOTE: This is the correlation-time augmenter used for contrastive pairs.
    It is distinct from training.training_base.GraphAugmenter, which is used for SSL pretraining.
    """

    @staticmethod
    def drop_edges(data: HeteroData, drop_prob: float = 0.1) -> HeteroData:
        data_aug = data.clone()
        for et in data_aug.edge_types:
            ei = data_aug[et].edge_index
            mask = torch.rand(ei.size(1)) > drop_prob
            data_aug[et].edge_index = ei[:, mask]
        return data_aug

    @staticmethod
    def mask_features(data: HeteroData, mask_prob: float = 0.1) -> HeteroData:
        data_aug = data.clone()
        if "alert" in data_aug:
            x = data_aug["alert"].x.clone()
            x[torch.rand(x.shape) < mask_prob] = 0.0
            data_aug["alert"].x = x
        return data_aug


# ============================================================================
# Module entrypoint
# ============================================================================

if __name__ == "__main__":
    print("MITRE-CORE HGNN Module v2.1")
    print("=" * 50)
    print("Usage:")
    print("  from hgnn.hgnn_correlation import HGNNCorrelationEngine")
    print("  # use_uf_refinement defaults to False (empirically validated, v2.6)")
    print("  engine = HGNNCorrelationEngine(")
    print("      model_path='hgnn_checkpoints/unsw_supervised.pt',")
    print("      confidence_gate=0.6,")
    print("      use_uf_refinement=False,  # default — do not change without re-running sweeps")
    print("  )")
    print("  result_df = engine.correlate(alert_dataframe)")
    print()
    print("Columns added to result_df:")
    print("  pred_cluster         — cluster ID (int)")
    print("  cluster_confidence   — HGNN max-softmax confidence [0,1]")
    print("  correlation_method   — 'hgnn' or 'hgnn+uf_refinement'")
