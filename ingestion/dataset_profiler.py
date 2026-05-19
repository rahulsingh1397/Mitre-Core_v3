"""
ingestion/dataset_profiler.py
Profile dataset structural properties and recommend checkpoint + hdbscan config.
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional

CHECKPOINT_REGISTRY = {
    "network_v9_v3":             "hgnn_checkpoints/network_v9_v3/network_it_best.pt",
    "multidomain_v2":            "hgnn_checkpoints/multidomain_v2/best_supervised.pt",
    "multidomain_v2_optc":       "hgnn_checkpoints/multidomain_v2_optc_finetuned/best_supervised.pt",
}

def profile_dataset(df: pd.DataFrame, sample_n: int = 1000) -> dict:
    """
    Profile dataset structure from a sample. Returns routing dict.
    Takes ~0.1s for 1000 rows.
    """
    sample = df.head(sample_n)
    n = len(sample)

    # IP structure
    src_ips = sample['src_ip'].nunique() if 'src_ip' in sample.columns else 0
    dst_ips = sample['dst_ip'].nunique() if 'dst_ip' in sample.columns else 0
    n_unique_ips = max(src_ips, dst_ips)
    ip_density = n_unique_ips / max(n, 1)

    # Other signals
    has_timestamps = any(c in sample.columns for c in ['timestamp', 'EndDate'])
    has_hostnames  = any(c in sample.columns for c in ['hostname', 'process_name', 'CommandLine'])
    has_labels     = 'campaign_id' in sample.columns and sample['campaign_id'].nunique() > 1

    # Decision tree
    if n_unique_ips == 0 and not has_hostnames:
        # NSL-KDD-like: no graph structure, feature-only
        checkpoint = "network_v9_v3"
        reason = "No IP/host structure → network_v9_v3 (MLP path, joint training benefit)"
    elif ip_density < 0.05 and has_timestamps:
        # Dense IoT / internal network (TON_IoT-like)
        checkpoint = "network_v9_v3"
        reason = f"Dense IP graph (density={ip_density:.3f}) → network_v9_v3"
    elif has_hostnames and n_unique_ips == 0:
        # Host/APT/Sysmon domain
        checkpoint = "multidomain_v2"
        reason = "Host domain (no IPs, has hostnames/processes) → multidomain_v2"
    else:
        # Enterprise sparse network or unrecognized — default to supervised
        checkpoint = "multidomain_v2"
        reason = f"Sparse/enterprise network (density={ip_density:.3f}) → multidomain_v2"

    return {
        "checkpoint_key":  checkpoint,
        "checkpoint_path": CHECKPOINT_REGISTRY[checkpoint],
        "ip_density":      ip_density,
        "n_unique_ips":    n_unique_ips,
        "has_timestamps":  has_timestamps,
        "has_hostnames":   has_hostnames,
        "has_labels":      has_labels,
        "reason":          reason,
    }


# ============================================================================
# Multi-Source Ingestion Pipeline (CS-4, Apr 28 2026)
# ============================================================================

MITRE_FORMAT_COLS = [
    "AlertId", "timestamp", "src_ip", "dst_ip", "hostname", "username",
    "alert_type", "tactic", "campaign_id", "protocol", "service",
    "src_bytes", "dst_bytes", "stage", "data_source",
]


class MultiSourceIngestionPipeline:
    """
    Combines alerts from N independent sensors/feeds into a single MITRE-format
    DataFrame, ready for AlertToGraphConverter.

    Usage
    -----
    pipeline = MultiSourceIngestionPipeline()
    pipeline.add_source(firewall_df,  source_name="palo_alto_fw")
    pipeline.add_source(edr_df,       source_name="crowdstrike_edr")
    pipeline.add_source(ids_df,       source_name="suricata_ids")
    merged = pipeline.merge()

    The merged DataFrame has a 'data_source' column tracking origin,
    deduplicated AlertIds (source_name prefix), and normalised column names.
    """

    # Columns required in each source DataFrame (or will be filled with NaN)
    # Supports both short aliases and full MITRE format names
    REQUIRED_COLS = ["timestamp", "src_ip", "alert_type", "tactic"]
    REQUIRED_COLS_ALT = ["EndDate", "SourceAddress", "MalwareIntelAttackType", "AttackTechnique"]

    def __init__(self, temporal_window_hours: float = 1.0):
        self._sources: list = []   # list of (df, source_name) tuples
        self.temporal_window = temporal_window_hours

    def add_source(self, df: pd.DataFrame, source_name: str) -> "MultiSourceIngestionPipeline":
        """Add one sensor feed. source_name becomes the data_source value."""
        # Check if DataFrame has either set of required columns
        has_standard = all(c in df.columns for c in self.REQUIRED_COLS)
        has_mitre = all(c in df.columns for c in self.REQUIRED_COLS_ALT)
        if not (has_standard or has_mitre):
            missing = [c for c in self.REQUIRED_COLS if c not in df.columns]
            raise ValueError(f"Source '{source_name}' missing required columns: {missing}")
        self._sources.append((df.copy(), source_name))
        return self

    def merge(self) -> pd.DataFrame:
        """
        Normalise + merge all sources.
        - Assigns unique AlertId per source (prefixed with source_name)
        - Adds 'data_source' column
        - Fills missing MITRE columns with sensible defaults
        - Returns single DataFrame sorted by timestamp
        """
        if not self._sources:
            raise ValueError("No sources added. Call add_source() first.")

        normalised = []
        for df, src_name in self._sources:
            ndf = df.copy()

            # 1. Assign AlertId if missing
            if "AlertId" not in ndf.columns:
                ndf["AlertId"] = [f"{src_name}_{i}" for i in range(len(ndf))]
            else:
                ndf["AlertId"] = src_name + "_" + ndf["AlertId"].astype(str)

            # 2. Stamp data_source
            ndf["data_source"] = src_name

            # 3. Fill missing MITRE columns
            for col in MITRE_FORMAT_COLS:
                if col not in ndf.columns:
                    ndf[col] = None

            # 4. Normalise column names (common aliases)
            _ALIASES = {
                "ip_src": "src_ip", "ip_dst": "dst_ip",
                "host": "hostname", "user": "username",
                "label": "alert_type", "attack_type": "tactic",
            }
            ndf = ndf.rename(columns={k: v for k, v in _ALIASES.items() if k in ndf.columns})

            # Select columns - ensure we only include each column once
            cols_to_use = list(dict.fromkeys(MITRE_FORMAT_COLS + [c for c in ndf.columns
                                                         if c not in MITRE_FORMAT_COLS]))
            normalised.append(ndf[cols_to_use])

        merged = pd.concat(normalised, ignore_index=True)

        # 5. Sort by timestamp
        try:
            merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
            merged = merged.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            pass

        return merged

    @staticmethod
    def from_csv_paths(paths: Dict[str, str], **kwargs) -> pd.DataFrame:
        """
        Convenience constructor.
        paths = {"palo_alto_fw": "data/fw.csv", "suricata": "data/ids.csv"}
        """
        pipeline = MultiSourceIngestionPipeline(**kwargs)
        for src_name, path in paths.items():
            df = pd.read_csv(path) if str(path).endswith(".csv") else pd.read_parquet(path)
            pipeline.add_source(df, source_name=src_name)
        return pipeline.merge()
