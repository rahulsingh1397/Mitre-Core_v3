# LEGACY: Multi-dataset training wrapper. Superseded by train_graph_mae_v9_multidata_fast.py.
# Kept for reference only.
# LEGACY: superseded by train_graph_mae_v9_multidata_fast.py.
"""
Train HGNN on Public Datasets
Trains the HGNN model using downloaded public cybersecurity datasets
"""

import os
import sys
import random
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mitre-core.train_hgnn")

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import warnings
warnings.filterwarnings('ignore')

# Import PyTorch Geometric
try:
    from torch_geometric.data import HeteroData
except ImportError:
    logger.error("torch_geometric not installed")
    sys.exit(1)

# Import HGNN modules
try:
    from hgnn.hgnn_correlation import (
        MITREHeteroGNN, AlertToGraphConverter, 
        HGNNCorrelationEngine, ContrastiveAlertLearner
    )
    from hgnn.hgnn_training import HGNNTrainer, AlertGraphDataset
    HGNN_AVAILABLE = True
except ImportError as e:
    logger.error(f"HGNN modules not available: {e}")
    HGNN_AVAILABLE = False
    sys.exit(1)


def apply_edge_dropout(graph, drop_rate=0.15):
    """Randomly drop edges from HeteroData graph to create augmented view."""
    g = graph.clone()
    # Get device from first edge_type's edge_index
    device = g[g.edge_types[0]].edge_index.device if g.edge_types else 'cpu'
    for edge_type in g.edge_types:
        n_edges = g[edge_type].edge_index.size(1)
        keep = torch.rand(n_edges, device=device) > drop_rate
        g[edge_type].edge_index = g[edge_type].edge_index[:, keep]
    return g


def apply_feature_augmentation(
    graph,
    mask_rate: float = 0.20,
    noise_std: float = 0.05,
):
    """
    Apply feature-level augmentation to alert node features.

    Two operations (both applied independently):
    1. Feature masking: randomly zero out `mask_rate` fraction of features
       (like Cutout/DropFeatures for tabular data)
    2. Gaussian noise: add small noise to continuous features (dims 2,3,10-12)
       which are already in [0,1] range (hour, dow, temporal densities)

    Args
    ----
    graph       : HeteroData to augment (cloned internally)
    mask_rate   : fraction of features to zero out per alert (default 0.20)
    noise_std   : std dev for Gaussian noise on continuous features (default 0.05)

    Returns cloned, augmented HeteroData. Original is not modified.
    """
    import torch
    g = graph.clone()
    if 'alert' not in g.node_types or g['alert'].x is None:
        return g

    x = g['alert'].x.clone()          # [N_alerts, 15]

    # 1. Random feature masking (all 15 features)
    if mask_rate > 0:
        mask = torch.rand_like(x) < mask_rate
        x[mask] = 0.0

    # 2. Gaussian noise on continuous features only (6-dim base features)
    #    Dim indices that are already in [0,1] range:
    #    2=hour/23, 3=dow/6 (normalized), 4=dst_port/65535, 5=src_port/65535
    if noise_std > 0:
        continuous_dims = [2, 3, 4, 5]
        noise = torch.randn(x.size(0), len(continuous_dims), device=x.device) * noise_std
        x[:, continuous_dims] = x[:, continuous_dims] + noise
        # Clamp to valid [0, 1+epsilon] range
        x[:, continuous_dims] = x[:, continuous_dims].clamp(0.0, 1.1)

    g['alert'].x = x
    return g


def apply_combined_augmentation(
    graph,
    edge_drop_rate: float = 0.15,
    feat_mask_rate: float = 0.20,
    feat_noise_std: float = 0.05,
):
    """Apply edge dropout + feature augmentation in one call."""
    g = apply_edge_dropout(graph, drop_rate=edge_drop_rate)
    g = apply_feature_augmentation(g, mask_rate=feat_mask_rate, noise_std=feat_noise_std)
    return g

class PublicDatasetGraphConverter:
    """
    DEPRECATED: Use hgnn.hgnn_correlation.AlertToGraphConverter instead.
    
    Converts public datasets in MITRE format to PyTorch Geometric HeteroData.
    Handles the converted column names from UNSW-NB15, CIC-IDS-2017, etc.
    """
    
    def __init__(self, temporal_window_hours: float = 1.0, build_bridge_edges: bool = True):
        self.temporal_window = temporal_window_hours
        self.build_bridge_edges = build_bridge_edges
        
    def convert(self, df: pd.DataFrame) -> HeteroData:
        """Convert MITRE-format DataFrame to heterogeneous graph."""
        from torch_geometric.data import HeteroData
        import torch
        from collections import defaultdict
        
        data = HeteroData()
        
        # Generate AlertId if not present
        if 'AlertId' not in df.columns:
            df = df.copy()
            df['AlertId'] = [f"alert_{i}" for i in range(len(df))]
        
        # Extract unique entities
        alerts = df['AlertId'].unique()
        
        # Users from username column
        if 'username' in df.columns:
            users = df['username'].dropna().unique()
        else:
            users = []
        
        # Hosts from hostname column
        if 'hostname' in df.columns:
            hosts = df['hostname'].dropna().unique()
        else:
            hosts = []
        
        # IPs from src_ip and dst_ip
        ips = []
        if 'src_ip' in df.columns:
            ips.extend(df['src_ip'].dropna().unique())
        if 'dst_ip' in df.columns:
            ips.extend(df['dst_ip'].dropna().unique())
        ips = list(set(ips))
        
        # Create node index mappings
        alert_to_idx = {a: i for i, a in enumerate(alerts)}
        user_to_idx = {u: i for i, u in enumerate(users)} if len(users) > 0 else {}
        host_to_idx = {h: i for i, h in enumerate(hosts)} if len(hosts) > 0 else {}
        ip_to_idx = {ip: i for i, ip in enumerate(ips)} if len(ips) > 0 else {}
        
        # Encode alert features (6-dim base; contextual features removed — see v5_contextual revert)
        data['alert'].x = self._encode_alert_features(df)
        
        # Encode entity features
        if len(users) > 0:
            data['user'].x = torch.ones(len(users), 1)
        if len(hosts) > 0:
            data['host'].x = torch.ones(len(hosts), 1)
        if len(ips) > 0:
            data['ip'].x = torch.ones(len(ips), 1)
        
        # Build edges
        edges = self._build_edges(df, alert_to_idx, user_to_idx, host_to_idx, ip_to_idx)
        
        # Add semantic similarity edges: same (tactic, service) within 2-hour window
        # Temporarily disabled for faster training
        if False and 'tactic' in df.columns and 'service' in df.columns:
            semantic_edges_src, semantic_edges_dst = [], []
            
            for i in range(len(df)):
                tactic_i = df.iloc[i].get('tactic', None)
                service_i = df.iloc[i].get('service', None)
                time_i = df.iloc[i].get('timestamp', None) or df.iloc[i].get('EndDate', None)
                
                for j in range(i + 1, min(len(df), i + 200)):  # local window only (O(n) not O(n²))
                    tactic_j = df.iloc[j].get('tactic', None)
                    service_j = df.iloc[j].get('service', None)
                    time_j = df.iloc[j].get('timestamp', None) or df.iloc[j].get('EndDate', None)
                    
                    if tactic_i == tactic_j and service_i == service_j:
                        # Check time proximity if timestamps available
                        if time_i is not None and time_j is not None:
                            time_diff_hrs = abs((pd.Timestamp(time_j) - pd.Timestamp(time_i)).total_seconds()) / 3600
                            if time_diff_hrs > 2.0:
                                continue
                        semantic_edges_src.extend([i, j])
                        semantic_edges_dst.extend([j, i])
            
            if semantic_edges_src:
                # Limit semantic similarity edges to prevent training slowdown
                max_semantic_edges = 1000
                if len(semantic_edges_src) > max_semantic_edges:
                    # Sample edges uniformly
                    indices = torch.randperm(len(semantic_edges_src) // 2)[:max_semantic_edges // 2]
                    new_src = []
                    new_dst = []
                    for idx in indices:
                        pos = idx.item() * 2
                        new_src.extend([semantic_edges_src[pos], semantic_edges_src[pos + 1]])
                        new_dst.extend([semantic_edges_dst[pos], semantic_edges_dst[pos + 1]])
                    semantic_edges_src = new_src
                    semantic_edges_dst = new_dst
                
                src = torch.tensor(semantic_edges_src, dtype=torch.long)
                dst = torch.tensor(semantic_edges_dst, dtype=torch.long)
                data[('alert', 'semantic_similar', 'alert')].edge_index = torch.stack([src, dst])
                logger.info(f"Added {len(semantic_edges_src)} semantic similarity edges")
        
        for edge_type, (src, dst) in edges.items():
            if len(src) > 0:
                data[edge_type].edge_index = torch.tensor([src, dst], dtype=torch.long)
        
        return data
    
    def _encode_alert_features(self, df: pd.DataFrame) -> np.ndarray:
        """Encode alert features to numeric vectors."""
        features = []
        
        # Tactic encoding
        if 'tactic' in df.columns:
            tactics = pd.Categorical(df['tactic']).codes
        else:
            tactics = np.zeros(len(df))
        
        # Alert type encoding (attack=1, normal=0)
        if 'alert_type' in df.columns:
            alert_types = (df['alert_type'] == 'attack').astype(int).values
        else:
            alert_types = np.zeros(len(df))
        
        # Temporal features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            hour = df['timestamp'].dt.hour.values
            day_of_week = df['timestamp'].dt.dayofweek.values
        else:
            hour = np.zeros(len(df))
            day_of_week = np.zeros(len(df))
        
        # Protocol encoding
        if 'protocol' in df.columns:
            protocols = pd.Categorical(df['protocol']).codes
        else:
            protocols = np.zeros(len(df))
        
        # Service encoding
        if 'service' in df.columns:
            services = pd.Categorical(df['service']).codes
        else:
            services = np.zeros(len(df))
        
        # Combine features - output 15 features to match network_v9_v3 expectation
        # Add 9 zero contextual features to reach 15 dimensions
        contextual_features = np.zeros((len(df), 9))  # 9 contextual features
        
        features = np.column_stack([
            tactics,
            alert_types,
            hour,  # Raw hour 0-23 for cyclical sin/cos encoding
            day_of_week,  # Raw day 0-6 for cyclical sin/cos encoding
            protocols,
            services,
            contextual_features,  # 9 zero contextual features
        ])

        return torch.tensor(features, dtype=torch.float)
    
    def _build_edges(self, df, alert_to_idx, user_to_idx, host_to_idx, ip_to_idx):
        """Build heterogeneous edges between nodes."""
        from collections import defaultdict
        edges = defaultdict(lambda: ([], []))
        
        # Add AlertId if missing
        if 'AlertId' not in df.columns:
            df = df.copy()
            df['AlertId'] = [f"alert_{i}" for i in range(len(df))]
        
        # Alert-to-Alert edges based on shared IPs
        ip_to_alerts = defaultdict(list)
        for idx, row in df.iterrows():
            alert_id = row['AlertId']
            if 'src_ip' in df.columns and pd.notna(row.get('src_ip')):
                ip_to_alerts[row['src_ip']].append(alert_to_idx[alert_id])
            if 'dst_ip' in df.columns and pd.notna(row.get('dst_ip')):
                ip_to_alerts[row['dst_ip']].append(alert_to_idx[alert_id])
        
        import random
        n_alerts = len(df)
        n_unique_ips = len(ip_to_alerts)
        ip_density = n_unique_ips / max(n_alerts, 1)  # fraction of unique IPs per alert

        # Dense dataset (few IPs, many alerts per IP → TON_IoT: ~0.01) → higher cap
        # Sparse dataset (many IPs, few alerts per IP → UNSW: ~0.35) → lower cap
        if ip_density < 0.05:          # Dense: TON_IoT, IoT datasets
            MAX_ALERTS_PER_IP = 80
            MAX_SHARES_IP_EDGES = 20000
        elif ip_density < 0.20:        # Medium: NSL-KDD (no IPs → density=0, handled separately)
            MAX_ALERTS_PER_IP = 40
            MAX_SHARES_IP_EDGES = 8000
        else:                           # Sparse: UNSW-NB15
            MAX_ALERTS_PER_IP = 20
            MAX_SHARES_IP_EDGES = 3000
            
        shares_src, shares_dst = [], []
        for ip, alert_indices in ip_to_alerts.items():
            if len(alert_indices) > MAX_ALERTS_PER_IP:
                alert_indices = random.sample(alert_indices, MAX_ALERTS_PER_IP)
            for i, alert_i in enumerate(alert_indices):
                for alert_j in alert_indices[i+1:]:
                    shares_src.extend([alert_i, alert_j])
                    shares_dst.extend([alert_j, alert_i])
        
        # Global cap: shuffle + truncate
        if len(shares_src) > MAX_SHARES_IP_EDGES * 2:
            perm = random.sample(range(0, len(shares_src), 2), MAX_SHARES_IP_EDGES)
            perm_flat = [p for i in perm for p in (i, i+1)]
            shares_src = [shares_src[p] for p in perm_flat]
            shares_dst = [shares_dst[p] for p in perm_flat]
            
        edges[('alert', 'shares_ip', 'alert')][0].extend(shares_src)
        edges[('alert', 'shares_ip', 'alert')][1].extend(shares_dst)
        
        # Alert-to-Alert edges based on temporal proximity (1-hour window)
        # Uses 'timestamp' column (UNSW-NB15, NSL-KDD, TON_IoT) or 'EndDate'
        time_col = None
        if 'EndDate' in df.columns:
            time_col = 'EndDate'
        elif 'timestamp' in df.columns:
            time_col = 'timestamp'

        if time_col is not None:
            times = pd.to_datetime(df[time_col], errors='coerce')
            alert_ids = df['AlertId'].tolist()
            valid_mask = times.notna()
            valid_times = times[valid_mask]
            valid_idxs = [alert_to_idx[alert_ids[i]] for i in range(len(df)) if valid_mask.iloc[i]]
            t_seconds = valid_times.values.astype('int64') / 1e9

            temp_src, temp_dst = [], []
            MAX_TEMPORAL_EDGES = 3000
            for i in range(len(valid_idxs)):
                for j in range(i + 1, min(i + 50, len(valid_idxs))):  # lookahead 50
                    if abs(t_seconds[j] - t_seconds[i]) <= 3600:  # 1-hour window
                        temp_src.extend([valid_idxs[i], valid_idxs[j]])
                        temp_dst.extend([valid_idxs[j], valid_idxs[i]])
                        if len(temp_src) >= MAX_TEMPORAL_EDGES * 2:
                            break
                if len(temp_src) >= MAX_TEMPORAL_EDGES * 2:
                    break
            if temp_src:
                edges[('alert', 'temporal_near', 'alert')][0].extend(temp_src)
                edges[('alert', 'temporal_near', 'alert')][1].extend(temp_dst)
        
        # Alert-to-User edges
        if 'username' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row.get('username')) and row['username'] in user_to_idx:
                    alert_idx = alert_to_idx[row['AlertId']]
                    user_idx = user_to_idx[row['username']]
                    edges[('user', 'owns', 'alert')][0].append(user_idx)
                    edges[('user', 'owns', 'alert')][1].append(alert_idx)
        
        # Alert-to-Host edges
        if 'hostname' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row.get('hostname')) and row['hostname'] in host_to_idx:
                    alert_idx = alert_to_idx[row['AlertId']]
                    host_idx = host_to_idx[row['hostname']]
                    edges[('host', 'generates', 'alert')][0].append(host_idx)
                    edges[('host', 'generates', 'alert')][1].append(alert_idx)
        
        # Cross-sensor bridge edges: IP resolves to hostname (mined from alerts with both)
        if self.build_bridge_edges:
            ip_to_host = {}
            for idx, row in df.iterrows():
                ip_val = row.get('src_ip') if pd.notna(row.get('src_ip')) else None
                host_val = row.get('hostname') if pd.notna(row.get('hostname')) else None
                if ip_val and host_val:
                    ip_to_host[ip_val] = host_val
            
            bridge_edges_added = 0
            for ip_val, host_val in ip_to_host.items():
                if ip_val in ip_to_idx and host_val in host_to_idx:
                    iid = ip_to_idx[ip_val]
                    hid = host_to_idx[host_val]
                    edges[('ip', 'resolves_to', 'host')][0].append(iid)
                    edges[('ip', 'resolves_to', 'host')][1].append(hid)
                    edges[('host', 'resolved_from', 'ip')][0].append(hid)
                    edges[('host', 'resolved_from', 'ip')][1].append(iid)
                    bridge_edges_added += 1
        
        return edges
    
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

        # Service frequency
        if 'service' in df.columns:
            svc_counts = df['service'].map(df['service'].value_counts())
            feats[:, 1] = torch.tensor(np.log1p(svc_counts.fillna(0).values) / denom, dtype=torch.float32)

        # dst_ip frequency
        if 'dst_ip' in df.columns:
            dst_counts = df['dst_ip'].map(df['dst_ip'].value_counts())
            feats[:, 2] = torch.tensor(np.log1p(dst_counts.fillna(0).values) / denom, dtype=torch.float32)

        # src_ip frequency
        if 'src_ip' in df.columns:
            src_counts = df['src_ip'].map(df['src_ip'].value_counts())
            feats[:, 3] = torch.tensor(np.log1p(src_counts.fillna(0).values) / denom, dtype=torch.float32)

        # Temporal density: O(n log n) with searchsorted
        if 'EndDate' in df.columns or 'timestamp' in df.columns:
            time_col = 'EndDate' if 'EndDate' in df.columns else 'timestamp'
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


class DatasetTrainer:
    """Train HGNN on downloaded public datasets."""
    
    def __init__(self, dataset_path: str = "./datasets", output_path: str = "./hgnn_checkpoints"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Use GPU if available for faster training
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def load_mitre_dataset(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load a dataset in MITRE-CORE format."""
        filepath = self.dataset_path / dataset_name / "mitre_format.csv"
        
        if not filepath.exists():
            logger.error(f"Dataset not found: {filepath}")
            return None
        
        logger.info(f"Loading {dataset_name} from {filepath}")
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} alerts")
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare training data with ground truth labels.
        
        For public datasets, we use attack category as campaign/cluster label.
        """
        # Filter out normal traffic for training (we want to cluster attacks)
        attack_df = df[df['alert_type'] == 'attack'].copy()
        
        if len(attack_df) == 0:
            logger.warning("No attack alerts found, using all data")
            attack_df = df.copy()
        
        logger.info(f"Using {len(attack_df)} alerts for training")
        
        # Group by campaign_id (ground truth clusters)
        # Each unique campaign_id represents a different attack campaign
        ground_truth = attack_df['campaign_id'].values
        
        # Map sparse labels to contiguous integers for CrossEntropyLoss
        unique_labels, contiguous_labels = np.unique(ground_truth, return_inverse=True)
        
        # Split into train/test
        train_df, test_df, train_labels, test_labels = train_test_split(
            attack_df, contiguous_labels, 
            test_size=test_size, 
            random_state=42,
            stratify=contiguous_labels  # Maintain class distribution
        )
        
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        logger.info(f"Train campaigns: {len(np.unique(train_labels))}")
        logger.info(f"Test campaigns: {len(np.unique(test_labels))}")
        
        return train_df, test_df, pd.Series(train_labels), pd.Series(test_labels)
    
    def train_on_dataset(self, dataset_name: str, epochs: int = 50, contrastive_epochs: int = 20, num_seeds: int = 5) -> Optional[str]:
        """Train HGNN on a specific dataset with multiple random seeds."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training on {dataset_name} with {num_seeds} random seeds")
        logger.info(f"{'='*60}")
        
        # Load data
        df = self.load_mitre_dataset(dataset_name)
        if df is None:
            return None
        
        # Prepare train/test split
        train_df, test_df, train_labels, test_labels = self.prepare_training_data(df)
        
        # Create graph datasets
        logger.info("\nConverting alerts to graphs...")
        
        # Use alert features for node encoding
        usernames = train_df.get('username', pd.Series(['unknown'] * len(train_df)))
        addresses = train_df.get('src_ip', pd.Series(['0.0.0.0'] * len(train_df)))
        
        # Build converter for public dataset format
        converter = PublicDatasetGraphConverter()
        
        # Convert to HeteroData graphs
        train_graphs = []
        train_labels_list = []
        
        # Group alerts into synthetic "campaigns" for training
        # We'll create mini-campaigns of 5-15 alerts each
        campaign_size = 10
        num_campaigns = len(train_df) // campaign_size
        
        logger.info(f"Creating {num_campaigns} mini-campaigns for training...")
        
        for i in range(0, min(len(train_df), num_campaigns * campaign_size), campaign_size):
            end_idx = min(i + campaign_size, len(train_df))
            mini_df = train_df.iloc[i:end_idx]
            mini_usernames = usernames.iloc[i:end_idx]
            mini_addresses = addresses.iloc[i:end_idx]
            
            # Build graph for this mini-campaign
            graph = converter.convert(mini_df)

            if graph is not None and 'alert' in graph.node_types:
                train_graphs.append(graph)
                # Use the most common campaign_id as label
                # Use mapped labels, not raw campaign IDs
                labels = train_labels.iloc[i:end_idx].values
                label = int(np.bincount(labels.astype(int)).argmax())
                train_labels_list.append(label)

        logger.info(f"Created {len(train_graphs)} training graphs")

        if len(train_graphs) == 0:
            logger.error("No valid training graphs created")
            return None

        # Create test graphs
        test_graphs = []
        test_labels_list = []

        for i in range(0, min(len(test_df), num_campaigns * campaign_size), campaign_size):
            end_idx = min(i + campaign_size, len(test_df))
            mini_df = test_df.iloc[i:end_idx]

            graph = converter.convert(mini_df)
            if graph is not None and 'alert' in graph.node_types:
                test_graphs.append(graph)
                labels = test_labels.iloc[i:end_idx].values
                label = int(np.bincount(labels.astype(int)).argmax())
                test_labels_list.append(label)

        logger.info(f"Created {len(test_graphs)} test graphs")

        # Ensure all graphs have consistent node types
        train_graphs = self._ensure_consistent_node_types(train_graphs)
        test_graphs = self._ensure_consistent_node_types(test_graphs)
        
        # Model config — detect real alert feature dim from data
        alert_feature_dim = 64
        for g in train_graphs:
            if 'alert' in g.node_types and g['alert'].x is not None:
                alert_feature_dim = g['alert'].x.shape[1]
                break
        hidden_dim = 128
        num_clusters = max(len(np.unique(np.concatenate([train_labels_list, test_labels_list]))), 10)
        
        # Run multiple seeds for robust statistics (M3: HGNN Single-Run Statistics)
        seed_accuracies = []
        best_overall_loss = float('inf')
        best_overall_model_path = None
        
        import random
        base_seeds = [42, 123, 456, 789, 999]
        seeds_to_run = base_seeds[:num_seeds] if num_seeds <= len(base_seeds) else [random.randint(1, 10000) for _ in range(num_seeds)]
        
        for seed_idx, seed in enumerate(seeds_to_run):
            logger.info(f"\n--- Running Seed {seed_idx+1}/{num_seeds} (Seed: {seed}) ---")
            
            # Set random seeds for reproducibility
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            
            logger.info(f"Model config: hidden_dim={hidden_dim}, num_clusters={num_clusters}")
            
            model = MITREHeteroGNN(
                alert_feature_dim=alert_feature_dim,
                hidden_dim=hidden_dim,
                num_clusters=num_clusters
            ).to(self.device)
            
            # Phase 1: Contrastive Pre-training
            logger.info(f"\nPhase 1: Contrastive Pre-training ({contrastive_epochs} epochs)")
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            from hgnn.cross_domain_contrastive import CrossDomainContrastiveLoss
            contrastive_fn = CrossDomainContrastiveLoss(temperature=0.3)
            
            best_seed_loss = float('inf')
            best_seed_model_state = None
            
            # Cross-graph batching: B mini-graphs per step so NT-Xent has real negatives.
            # Within-graph augmented pairs = positives; cross-graph pairs = negatives.
            # Without this, each step has only ~50 alerts with no negatives → loss→0.
            BATCH_SIZE = 16

            for epoch in range(contrastive_epochs):
                model.train()
                total_loss = 0
                n_steps = 0

                graphs_subset = train_graphs[:1000]
                random.shuffle(graphs_subset)

                for batch_start in range(0, len(graphs_subset), BATCH_SIZE):
                    batch = graphs_subset[batch_start:batch_start + BATCH_SIZE]
                    if len(batch) < 2:
                        continue

                    optimizer.zero_grad()

                    z1_parts, z2_parts = [], []
                    for graph in batch:
                        aug1 = apply_edge_dropout(graph, drop_rate=0.15).to(self.device)
                        aug2 = apply_edge_dropout(graph, drop_rate=0.15).to(self.device)
                        _, x1 = model(aug1)
                        _, x2 = model(aug2)
                        if 'alert' in x1 and 'alert' in x2:
                            z1_parts.append(x1['alert'])
                            z2_parts.append(x2['alert'])

                    if not z1_parts:
                        continue

                    # Concatenate across graphs: [B*N_alerts, 128]
                    # Cross-graph pairs are negatives; same-graph augmented pairs are positives
                    z1 = torch.cat(z1_parts, dim=0)
                    z2 = torch.cat(z2_parts, dim=0)

                    loss = contrastive_fn(z1, z2, "train", "train")
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_steps += 1

                avg_loss = total_loss / max(n_steps, 1)
                
                if avg_loss < best_seed_loss:
                    best_seed_loss = avg_loss
                    best_seed_model_state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_seed_loss,
                        'num_clusters': num_clusters,
                        'hidden_dim': hidden_dim,
                        'seed': seed
                    }
                
                if (epoch + 1) % 5 == 0:
                    logger.info(f"Epoch {epoch+1}/{contrastive_epochs}, Loss: {avg_loss:.4f}")
            
            # Phase 2 removed — pure unsupervised training now
            
            # Save the best overall model across all seeds
            if best_seed_loss < best_overall_loss:
                best_overall_loss = best_seed_loss
                best_overall_model_path = self.output_path / f"{dataset_name}_best.pt"
                if best_seed_model_state:
                    torch.save(best_seed_model_state, best_overall_model_path)
                
            # Load best model for this seed to evaluate (optional, mainly for stats logging)
            if best_seed_model_state:
                model.load_state_dict(best_seed_model_state['model_state_dict'])
                
            # Since we removed supervised fine-tuning, accuracy is no longer meaningful here.
            # Just recording a dummy value to keep stats logging intact.
            seed_accuracies.append(0.0)
            logger.info(f"Seed {seed} unsupervised pre-training complete.")
            
            # --- Baseline Homogeneous GNN Training & Evaluation ---
            if seed_idx == 0:  # Only run baseline once per dataset
                logger.info(f"\n{'='*60}")
                logger.info(f"Baseline Comparison: Skipped (no supervised baseline)")
                logger.info(f"{'='*60}")
                self.baseline_acc = 0.0
            
        # Compute and log multi-seed statistics
        mean_acc = np.mean(seed_accuracies)
        std_acc = np.std(seed_accuracies)
        logger.info(f"\n{'='*60}")
        logger.info(f"Multi-Seed Statistics for {dataset_name} ({num_seeds} runs)")
        logger.info(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        logger.info(f"Accuracies across seeds: {[f'{acc:.4f}' for acc in seed_accuracies]}")
        logger.info(f"Best overall model saved to: {best_overall_model_path} (Loss: {best_overall_loss:.4f})")
        logger.info(f"{'='*60}\n")
        
        # Save statistics to file for reporting
        stats_path = self.output_path / f"{dataset_name}_hgnn_stats.json"
        import json
        with open(stats_path, 'w') as f:
            json.dump({
                "dataset": dataset_name,
                "num_seeds": num_seeds,
                "mean_accuracy": float(mean_acc),
                "std_accuracy": float(std_acc),
                "seed_accuracies": [float(acc) for acc in seed_accuracies],
                "seeds_used": seeds_to_run,
                "baseline_accuracy": float(getattr(self, 'baseline_acc', 0.0)),
                "improvement_over_baseline": float(mean_acc - getattr(self, 'baseline_acc', 0.0))
            }, f, indent=4)
        
        return str(best_overall_model_path)
    
    def _ensure_consistent_node_types(self, graphs: List[HeteroData]) -> List[HeteroData]:
        """Simplified: Keep alert nodes and create minimal edges if needed."""
        import torch
        
        simplified_graphs = []
        for graph in graphs:
            # Check if alert node type exists
            if 'alert' not in graph.node_types:
                continue
                
            num_alerts = graph['alert'].x.shape[0]
            
            # Create minimal graph with only alert nodes
            new_graph = HeteroData()
            new_graph['alert'].x = graph['alert'].x
            
            # Copy alert-to-alert edges if they exist
            has_edges = False
            for edge_type in graph.edge_types:
                src, rel, dst = edge_type
                if src == 'alert' and dst == 'alert':
                    edge_index = graph[edge_type].edge_index
                    if edge_index.numel() > 0 and edge_index.max() < num_alerts:
                        new_graph[edge_type].edge_index = edge_index
                        has_edges = True
            
            # If no alert-to-alert edges, create self-loops so GNN can work
            if not has_edges:
                # Create self-loop edges for each alert
                self_loops = torch.arange(num_alerts, dtype=torch.long).unsqueeze(0).repeat(2, 1)
                new_graph[('alert', 'self_loop', 'alert')].edge_index = self_loops
            
            simplified_graphs.append(new_graph)
        
        logger.info(f"Simplified {len(simplified_graphs)} graphs to alert-only")
        return simplified_graphs
    
    def evaluate_model(self, model, test_graphs, test_labels):
        """Evaluate trained model on test set."""
        logger.info(f"\n{'='*60}")
        logger.info("Evaluation on Test Set")
        logger.info(f"{'='*60}")
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for graph, true_label in zip(test_graphs, test_labels):
                graph = graph.to(self.device)
                cluster_logits, _ = model(graph)
                
                # Majority vote prediction
                predictions = torch.argmax(cluster_logits, dim=-1)
                pred_label = torch.mode(predictions).values.item()
                
                if pred_label == true_label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")
        
        return accuracy
    
    def train_all_datasets(self, epochs: int = 50, contrastive_epochs: int = 20, num_seeds: int = 5):
        """Train on all available datasets."""
        available_datasets = []
        
        # Use filtered datasets if set via self.datasets, otherwise auto-detect
        candidate_names = list(getattr(self, 'datasets', {}).keys()) or \
            ['nsl_kdd', 'unsw_nb15', 'cicids2017', 'cicids2018']
        for dataset_name in candidate_names:
            filepath = self.dataset_path / dataset_name / "mitre_format.csv"
            if filepath.exists():
                available_datasets.append(dataset_name)
        
        if not available_datasets:
            logger.error("No datasets found. Run download_datasets.py first.")
            return
        
        logger.info(f"Found datasets: {available_datasets}")
        
        trained_models = {}
        
        for dataset_name in available_datasets:
            model_path = self.train_on_dataset(dataset_name, epochs=epochs,
                                               contrastive_epochs=contrastive_epochs,
                                               num_seeds=num_seeds)
            if model_path:
                trained_models[dataset_name] = model_path
        
        logger.info(f"\n{'='*60}")
        logger.info("Training Summary")
        logger.info(f"{'='*60}")
        for dataset, path in trained_models.items():
            logger.info(f"✓ {dataset}: {path}")
        
        return trained_models


def main():
    """Main training entry point."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--contrastive_epochs', type=int, default=20)
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Run only these datasets (e.g. unsw_nb15 cicids2017)')
    parser.add_argument('--output_dir', type=str, default='./hgnn_checkpoints',
                        help='Directory to save checkpoints')
    args = parser.parse_args()

    trainer = DatasetTrainer(output_path=args.output_dir)
    if args.datasets:
        trainer.datasets = {d: d for d in args.datasets}
    trained_models = trainer.train_all_datasets(
        epochs=args.epochs,
        contrastive_epochs=args.contrastive_epochs,
        num_seeds=args.num_seeds
    )
    
    if trained_models:
        logger.info(f"\n{'='*60}")
        logger.info("All models trained successfully!")
        logger.info(f"Models saved to: {trainer.output_path}")
        logger.info(f"{'='*60}")
    else:
        logger.error("Training failed. Check dataset availability.")


if __name__ == "__main__":
    main()
