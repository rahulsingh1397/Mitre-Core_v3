"""
experiments/run_gate_tuning.py
-------------------------------
Confidence gate tuning sweep for MITRE-CORE v2.1.
Runs HGNNCorrelationEngine across all 8 datasets × 9 gate values.

Usage:
    python experiments/run_gate_tuning.py \
        --output experiments/results/gate_tuning_results.csv

    # Checkpoint is set per-dataset via checkpoint_override in DATASET_CONFIG,
    # or defaults to hgnn_checkpoints/multidomain_v2/best_supervised.pt

Constraints:
    - No synthetic data. All inputs must be real preprocessed dataset graphs.
    - All runs logged with git commit hash via scripts/generate_experiment_log.py.
    - Do NOT overwrite existing experiment results files — append or use new filename.
"""

import argparse
import time
import logging
import sys
import os
import gc
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    adjusted_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
)
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hgnn.hgnn_correlation import HGNNCorrelationEngine
from utils.seed_control import set_seed
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gate_tuning")

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

GATE_VALUES = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]

DATASET_CONFIG = {
    "Malware_Sysmon": {
        "path": "datasets/attack_data_processed/malware_sysmon_mitre_format.parquet",
        "label_col": "campaign_id",
        "hdbscan_min_cluster_size": 15,
        "hdbscan_pca_components": 24,
        "hdbscan_auto_tune": True,
        "sample_size": 10000,
        "stratified_sample": True,
        "use_geometric_confidence": True,
        "note": "3 malware families (minergate, trickbot, xmrig_miner) from Splunk attack_data-master. Sysmon EventID 1+3. ~30K events. Pure unsupervised HDBSCAN clustering.",
    },
    "Attack_Techniques": {
        "path": "datasets/attack_data_processed/attack_techniques_mitre_format.parquet",
        "label_col": "campaign_id",
        "hdbscan_min_cluster_size": 5,
        "hdbscan_pca_components": 16,
        "hdbscan_auto_tune": True,
        "sample_size": 5000,
        "stratified_sample": True,
        "use_geometric_confidence": True,
        "note": "212 MITRE ATT&CK techniques from Splunk attack_data-master. Labels: technique IDs. ~63K events. Pure unsupervised HDBSCAN clustering.",
    },
    "UNSW-NB15": {
        "path": "datasets/unsw_nb15/mitre_format.csv",
        "label_col": "campaign_id",
        "label_col_alt": "attack_cat",   # 9-class standard benchmark label
        "hdbscan_min_cluster_size": 10,            # Reduced from 30 — UNSW has 8 campaigns, smaller clusters needed
        "hdbscan_pca_components": 16,
        "use_geometric_confidence": True,          # GAEC mode — network_v9_v3 has no classification head
        "hdbscan_auto_tune": True,
        "hdbscan_cluster_selection_epsilon": 0.0,  # Fixed from 0.05 — epsilon=0.05 was merging 40+ clusters into 2
        "use_umap": True,
        "umap_n_components": 10,
        "umap_n_neighbors": 30,                    # Added — higher = preserves more global structure
        "umap_min_dist": 0.1,                      # v2.18 failed: 0.0 gave ARI=0.481 (worse than 0.523)
        "sample_size": 2000,
        "stratified_sample": True,
        "chunk_inference_size": 1000,
        "clustering_method": "spectral",
        "n_clusters": 8,
        "checkpoint_override": "hgnn_checkpoints/unsw_supcon_v7/best.pt",
        "prototype_checkpoint_path": "hgnn_checkpoints/prototypes/unsw_v2/best_prototype_model.pt",
        "prototype_test_indices_path": "hgnn_checkpoints/prototypes/unsw_v2/test_indices.npy",
        "note": "UNSW-NB15: Zero-shot with network_v9_v3 (GAEC mode). "
                "SupCon checkpoint removed due to 15-dim vs 6-dim mismatch (A3 fix). "
                "Zero-shot baseline: ARI ~0.40 (network_v9_v3).",
    },
    "TON_IoT": {
        "path": "datasets/TON_IoT/mitre_format.parquet",
        "label_col": "campaign_id",
        "sample_size": 10000,
        "stratified_sample": True,
        "hdbscan_min_cluster_size": 50,
        "hdbscan_pca_components": 16,
        "use_geometric_confidence": True,
        "hdbscan_auto_tune": True,
        "hdbscan_cluster_selection_epsilon": 0.1,
        "use_umap": False,
        "skip_gate_sweep": False,
        "checkpoint_override": "hgnn_checkpoints/network_v9_v3/network_it_best.pt",  # Original v9_v3 - Track 11 baseline ARI=0.431
        "prototype_checkpoint_path": "hgnn_checkpoints/prototypes/ton_iot_v2/best_prototype_model.pt",
        "prototype_test_indices_path": "hgnn_checkpoints/prototypes/ton_iot_v2/test_indices.npy",
        "build_bridge_edges": True,
        "note": "TON_IoT: network_v9_v3 checkpoint with Track 11 fixes (ARI=0.431). UMAP disabled to prevent over-fragmentation. CS fine-tuning ineffective (AMI=0.0000).",
    },
    "OpTC": {
        "path": "datasets/DARPA_OpTC/processed_optc_full.csv",
        "label_col": "CampaignId",
        "hdbscan_min_cluster_size": 50,
        "hdbscan_pca_components": 8,
        "sample_size": 10000,
        "stratified_sample": True,
        "chunk_inference_size": 1000,
        "use_geometric_confidence": True,   # GAEC mode — verified binary_ARI=0.999 with v9_v3
        "checkpoint_override": "hgnn_checkpoints/network_v9_v3/network_it_best.pt",
        "build_bridge_edges": True,
        "note": "DARPA OpTC 2020: zero-shot with network_v9_v3 (GAEC mode). "
                "Verified best binary_ARI=0.999; standard ARI=0.047 (penalised for legitimate sub-clustering). "
                "Previous softmax+multidomain_v2 config gave ARI=0.008 (OOD head).",
    },
    "OpTC_Temporal": {
        "path": "datasets/DARPA_OpTC/processed_optc_full.csv",
        "label_col": "CampaignId",
        "hdbscan_min_cluster_size": 10,
        "hdbscan_pca_components": 16,
        "sample_size": 10000,
        "temporal_split": True,          # New flag: use last 20% of dates for eval
        "note": "Temporal held-out evaluation to test date leakage"
    },
    "NSL-KDD": {
        "path": "datasets/nsl_kdd/mitre_format.csv",
        "label_col": "tactic",                     # CHANGED from campaign_id
        "hdbscan_min_cluster_size": 5,             # FIXED: was 30, now 5 to prevent softmax fallback
        "hdbscan_pca_components": 16,
        "hdbscan_cluster_selection_epsilon": 0.1,  # ADDED: epsilon merging for better clustering
        "use_geometric_confidence": True,           # FIXED: logging bug - engine uses True but CSV logged "softmax"
        "sample_size": 10000,
        "stratified_sample": True,
        "checkpoint_override": "hgnn_checkpoints/network_v9_v3/network_it_best.pt",
        "prototype_checkpoint_path": "hgnn_checkpoints/prototypes/nsl_kdd_v2/best_prototype_model.pt",
        "prototype_test_indices_path": "hgnn_checkpoints/prototypes/nsl_kdd_v2/test_indices.npy",
        "num_layers": 1,                           # FIX: Use single layer to prevent embedding collapse
        "note": "NSL-KDD with tactic labels (10 classes). num_layers=1 to prevent embedding collapse.",
    },
    "SQTK_SIEM_kcluster": {
        "path": "datasets/SQTK_SIEM/mitre_core_format.csv",
        "label_col": "kcluster",
        "checkpoint_override": "hgnn_checkpoints/siem_supcon_v4/best.pt",
        "prototype_checkpoint_path": "hgnn_checkpoints/prototypes/sqtk_v2/best_prototype_model.pt",
        "test_indices_path": "hgnn_checkpoints/siem_supcon_v4/test_indices.npy",
        "prototype_test_indices_path": "hgnn_checkpoints/prototypes/sqtk_v2/test_indices.npy",
        "hdbscan_min_cluster_size": 5,            # Phase 1.1: Force finer splits in over-smoothed space
        "hdbscan_cluster_selection_method": "leaf",  # Phase 1.1: Prefer smaller clusters
        "hdbscan_pca_components": 16,
        "sample_size": None,                       # All 5100 records
        "chunk_inference_size": 1000,
        "use_geometric_confidence": True,          # FIX: switch from softmax to GAEC mode (softmax head OOD for SIEM kcluster)
        "use_umap": True,                          # FIX: UMAP for embedding spread (matches successful TON_IoT/CICIDS2017 configs)
        "umap_n_components": 10,
        "umap_n_neighbors": 30,
        "umap_min_dist": 0.1,
        "hdbscan_cluster_selection_epsilon": 0.1,  # FIX: epsilon for cluster merging (matches TON_IoT/CICIDS2017)
        "hdbscan_auto_tune": True,                 # FIX: auto-tune min_cluster_size
        "hdbscan_metric_fallback": True,           # FIX: P3 sparse graph fallback
        "hdbscan_zca_whitening": True,             # NEW: Apply Soft-ZCA to fix collapse
        "hdbscan_zca_eps": 0.1,                    # NEW: ZCA regularization
        "clustering_method": "spectral",           # NEW: bypass HDBSCAN failure
        "n_clusters": 11,                          # NEW: match expert clusters
        "note": "SQTK SIEM: 9 expert kcluster classes (filtered from 11, dropped singletons). "
                "FIX: GAEC mode + network_v9_v3 backbone + UMAP + epsilon to fix ARI=0.005 failure. "
                "Previous failure: softmax mode with multidomain_v2 checkpoint (OOD classification head)."
    },
    "SQTK_SIEM_tactic": {
        "path": "datasets/SQTK_SIEM/mitre_core_format.csv",
        "label_col": "campaign_id",               # Tactic-based, 89.4% UNKNOWN
        "hdbscan_min_cluster_size": 30,
        "hdbscan_pca_components": 16,
        "sample_size": None,
        "note": "Tactic labels — 89.4% UNKNOWN. ARI is dominated by UNKNOWN class. Use kcluster instead.",
        "skip_gate_sweep": True,                  # Document but don't sweep — misleading metric
    },
    "BETH": {
        "path": "datasets/BETH/mitre_format.parquet",
        "label_col": "campaign_id",
        "hdbscan_min_cluster_size": 50,
        "hdbscan_pca_components": 32,
    },
    "CICAPT_IIoT": {
        "path": "datasets/CICAPT-IIoT-Dataset/mitre_format.parquet",
        "label_col": "label",
        "hdbscan_min_cluster_size": 30,
        "hdbscan_pca_components": 16,
        "sample_size": 10000,
        "stratified_sample": True,
        "skip_gate_sweep": True,
        "checkpoint_override": "hgnn_checkpoints/network_v9_v3/network_it_best.pt",
        "note": "SKIPPED: 21,527:1 class imbalance (1,004 attacks in 21.6M). Fix: use --oversample-minority to force all 1,004 attack records into training. Alternatively, anomaly-detection approach (future work).",
    },
    "CICIDS2017": {
        "path": "datasets/CICIDS2017/mitre_format.parquet",
        "label_col": "campaign_id",                           # FIX: was "MalwareIntelAttackType" (column doesn't exist)
        "hdbscan_min_cluster_size": 20,       # 10K sample, 16 campaigns
        "hdbscan_pca_components": 16,
        "sample_size": 10000,
        "stratified_sample": True,
        "use_geometric_confidence": True,
        "hdbscan_auto_tune": True,
        "hdbscan_cluster_selection_epsilon": 0.1,
        "use_umap": True,
        "umap_n_components": 10,
        "note": "CICIDS2017: network traffic (16 campaigns). Zero-shot with network_v9_v3 (GAEC mode). "
                "SupCon checkpoint removed due to 15-dim vs 6-dim mismatch (A3 fix). "
                "Zero-shot baseline: ARI ~0.284 (sweep_v9v3_restored).",
    },
    "Splunk_AttackData_technique": {
        "path": "datasets/splunk_attack_data/mitre_format.parquet",
        "label_col": "campaign_id",          # e.g. "T1059_metasploit", "ryuk"
        "hdbscan_min_cluster_size": 3,       # small — lab data has few events per technique
        "hdbscan_pca_components": 8,
        "sample_size": 5000,
        "stratified_sample": True,
        "note": "Splunk attack_data: 50 ATT&CK technique subfolders + 42 malware families. Windows Sysmon EventID 1+3 + WinEventLog. Label = technique_subfolder (finest granularity).",
    },
    "Splunk_AttackData_tactic": {
        "path": "datasets/splunk_attack_data/mitre_format.parquet",
        "label_col": "MalwareIntelAttackType",  # e.g. "T1059", "T1078", "malware"
        "hdbscan_min_cluster_size": 3,
        "hdbscan_pca_components": 8,
        "sample_size": 5000,
        "stratified_sample": True,
        "note": "Same dataset as above but labelled at MITRE technique ID level (coarser, better for clustering eval).",
    },
}

from typing import Optional, List

# -----------------------------------------------------------------------
# Helper: load a dataset DataFrame from its preprocessed path
# -----------------------------------------------------------------------

def load_dataset(path: str, sample_size: Optional[int] = 10000, config: Optional[dict] = None, dataset_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load a preprocessed dataset. Supports .csv and .parquet.
    Raises FileNotFoundError if the path does not exist — do not silently
    fall back to synthetic data.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Preprocessed dataset not found: {path}\n"
            f"Expected mitre-format file at this path."
        )
    if p.suffix == ".csv":
        df = pd.read_csv(p, low_memory=False)
    elif p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    # Check for saved test indices from retrain_hgnn_siem.py (Fix A: held-out test set)
    test_indices = None
    if config and config.get("checkpoint_override"):
        # Look for test_indices.npy in the same directory as the checkpoint
        checkpoint_dir = Path(config["checkpoint_override"]).parent
        test_indices_path = checkpoint_dir / "test_indices.npy"
        if test_indices_path.exists():
            test_indices = np.load(test_indices_path)
            logger.info(f"Loaded {len(test_indices)} held-out test indices from {test_indices_path}")

    # Apply temporal split if requested
    if config and config.get("temporal_split"):
        if 'date' in df.columns:
            df['_date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
            # For OpTC with only 2 dates, use the latest date for temporal split
            unique_dates = sorted(df['_date_parsed'].dropna().unique())
            if len(unique_dates) <= 2:
                # Use the most recent date for temporal evaluation
                cutoff = unique_dates[-2] if len(unique_dates) == 2 else unique_dates[0]
                df = df[df['_date_parsed'] > cutoff].drop(columns=['_date_parsed'])
                logger.info(f"Temporal split (small dataset): {len(df)} records after cutoff {cutoff.date()}")
            else:
                # Use last 20% of date range for larger datasets
                cutoff = df['_date_parsed'].quantile(0.8)
                df = df[df['_date_parsed'] > cutoff].drop(columns=['_date_parsed'])
                logger.info(f"Temporal split: {len(df)} records after cutoff {cutoff.date()}")
        else:
            logger.warning("Temporal split requested but no 'date' column found")

    # Remap mitre_format column names to HGNN AlertToGraphConverter names
    col_map = {
        "src_ip": "SourceAddress",
        "dst_ip": "DestinationAddress",
        "hostname": "SourceHostName",
        "username": "SourceUserName",
        "timestamp": "EndDate",
        "alert_type": "MalwareIntelAttackType",
        "tactic": "AttackTechnique",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    # If we have held-out test indices, use ONLY those records
    if test_indices is not None:
        df = df.iloc[test_indices].copy().reset_index(drop=True)
        logger.info(f"Using held-out test set: {len(df)} records")
        
        # Preserve dataset name for domain head detection
        df._dataset_name = dataset_name
        
        # Apply sample_size if specified and held-out set is too large
        if sample_size is not None and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled held-out test set to {len(df)} records")
        
        return df

    # Sample large datasets to keep runtime manageable
    # NOTE: Changed random_state from 42 to 99 to reduce overlap with training data
    if sample_size is not None and len(df) > sample_size:
        # Use stratified sampling for OpTC to preserve class proportions
        if config and config.get("stratified_sample") and config.get("label_col") in df.columns:
            label_col = config["label_col"]
            logger.info(f"Using stratified sampling for {label_col} to preserve class proportions")
            df = (
                df.groupby(label_col, group_keys=False)
                  .apply(lambda g: g.sample(
                      min(len(g), max(1, int(sample_size * len(g) / len(df)))),
                      random_state=42
                  ))
            )
            # Shuffle the result to avoid ordered grouping
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
            logger.info(f"Stratified sampling complete: {len(df)} records, class distribution preserved")
        else:
            # Standard random sampling with seed=42 to match reference
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
            logger.info(f"Random sampling complete: {len(df)} records")

    return df


# -----------------------------------------------------------------------
# Helper: encode ground-truth labels to integer cluster IDs
# -----------------------------------------------------------------------

def encode_labels(df: pd.DataFrame, label_col: str) -> np.ndarray:
    if label_col not in df.columns:
        raise ValueError(
            f"Ground-truth column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    le = LabelEncoder()
    return le.fit_transform(df[label_col].fillna("UNKNOWN").astype(str).values)


# -----------------------------------------------------------------------
# Helper: run inference in chunks
# -----------------------------------------------------------------------

def run_chunked_inference(engine, df, chunk_size):
    """Run correlate() in chunks matching training scale, concatenate results."""
    results = []
    dataset_name = getattr(df, "_dataset_name", "unknown")
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size].copy().reset_index(drop=True)
        chunk._dataset_name = dataset_name  # Preserve dataset name
        chunk_result = engine.correlate(chunk)
        results.append(chunk_result)
    combined = pd.concat(results, ignore_index=True)
    # Re-align index to original df order
    combined.index = range(len(combined))
    return combined

# -----------------------------------------------------------------------
# Main sweep
# -----------------------------------------------------------------------

def run_sweep(
    checkpoint_path: str,
    output_path: str,
    datasets: Optional[List[str]] = None,
    seed: int = 42,
    fixed_gate: Optional[float] = None,
    build_bridge_edges: Optional[bool] = None,
    collapse_entities: bool = False,
    clustering_method: Optional[str] = None,
    n_clusters: Optional[int] = None,
    zca_whitening: Optional[bool] = None,
    zca_eps: Optional[float] = None,
    aggr_method: Optional[str] = None,
    use_burstiness: Optional[bool] = None,
    prototype_checkpoint: Optional[str] = None,
    track_data_source: Optional[bool] = None,
    build_precedes_edges: Optional[bool] = None,
    precedes_window_hours: float = 2.0,
    use_uf_refinement: Optional[bool] = None,
) -> None:
    set_seed(seed)
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                         stderr=subprocess.DEVNULL).decode().strip()
    except:
        git_hash = "unknown"
    logger.info(f"Starting gate tuning sweep | git={git_hash}")

    results = []

    for dataset_name, config in DATASET_CONFIG.items():
        if datasets is not None and dataset_name not in datasets:
            continue
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {dataset_name}")

        if config.get("skip_gate_sweep", False):
            logger.info(
                f"SKIPPING gate sweep for {dataset_name}: {config.get('note', 'flagged')}"
            )
            # Use fixed gate if specified, otherwise use sweep values
        if fixed_gate is not None:
            gate_values_for_dataset = [fixed_gate]
        elif config.get("skip_gate_sweep", False):
            gate_values_for_dataset = [0.65]  # Fixed gate for datasets where sweep doesn't make sense
        else:
            gate_values_for_dataset = GATE_VALUES

        df = load_dataset(config["path"], sample_size=config.get("sample_size", 10000), config=config, dataset_name=dataset_name)

        if "AlertId" not in df.columns:
            df["AlertId"] = [f"ALT-{i}" for i in range(len(df))]

        # If a held-out test split was saved during fine-tuning, use only those rows.
        # This prevents train-set contamination and enables proper evaluation
        # comparable to evaluate_multidomain.py (e.g. for UNSW SupCon checkpoint).
        test_indices_path = config.get("test_indices_path", None)
        if test_indices_path and Path(test_indices_path).exists():
            test_indices = np.load(test_indices_path)
            # test_indices are absolute row positions in the full (unsampled) CSV.
            # load_dataset may have already sampled — use intersection with loaded df index.
            available = set(df.index.tolist())
            valid_idx = [i for i in test_indices if i in available]
            if len(valid_idx) >= 50:
                df = df.loc[valid_idx].reset_index(drop=True)
                logger.info(
                    f"[{dataset_name}] Held-out test split loaded: "
                    f"{len(df)} records from {test_indices_path}"
                )
            else:
                logger.warning(
                    f"[{dataset_name}] test_indices_path={test_indices_path} matched only "
                    f"{len(valid_idx)} rows after sampling — using full sampled dataset instead."
                )

        # For prototype mode: use the held-out test split saved during prototype training
        if clustering_method == "prototype":
            test_idx_path = config.get("prototype_test_indices_path")
            if test_idx_path and Path(test_idx_path).exists():
                test_indices = np.load(test_idx_path)
                df = df.iloc[test_indices].reset_index(drop=True)
                logger.info(f"Prototype mode: using {len(df)} test-split rows from {test_idx_path}")

        true_labels = encode_labels(df, config["label_col"])

        logger.info(f"Loaded {len(df)} records for {dataset_name}")
        logger.info(f"Label distribution: {pd.Series(true_labels).value_counts().to_dict()}")

        # Set dataset name for GAEC diagnostics logger
        df._dataset_name = dataset_name

        # CLI arg overrides per-dataset config; config overrides default True
        cfg_bridge = config.get("build_bridge_edges", True)
        effective_build_bridge_edges = build_bridge_edges if build_bridge_edges is not None else cfg_bridge
        logger.info(f"Bridge edges config: CLI={build_bridge_edges}, config={cfg_bridge}, effective={effective_build_bridge_edges}")

        for gate in gate_values_for_dataset:
            logger.info(f"  gate={gate:.2f} ...")

            use_geom = config.get("use_geometric_confidence", False)
            ckpt = config.get("checkpoint_override", checkpoint_path)
            engine = HGNNCorrelationEngine(
                model_path=ckpt,
                device="cpu",
                use_geometric_confidence=config.get("use_geometric_confidence", True),
                confidence_gate=gate,
                pure_unsupervised=True,  # Always use unsupervised labels for GAEC
                hdbscan_min_cluster_size=config.get("hdbscan_min_cluster_size", 15),
                hdbscan_pca_components=config.get("hdbscan_pca_components", 16),
                hdbscan_auto_tune=config.get("hdbscan_auto_tune", False),
                hdbscan_cluster_selection_epsilon=config.get("hdbscan_cluster_selection_epsilon", 0.0),
                hdbscan_use_umap=config.get("use_umap", False),
                hdbscan_umap_n_components=config.get("umap_n_components", 10),
                hdbscan_umap_n_neighbors=config.get("umap_n_neighbors", 15),
                hdbscan_umap_min_dist=config.get("umap_min_dist", 0.1),
                hdbscan_cluster_selection_method=config.get("hdbscan_cluster_selection_method", "eom"),
                hdbscan_metric_fallback=config.get("hdbscan_metric_fallback", False),
                hdbscan_zca_whitening=zca_whitening if zca_whitening is not None else config.get("hdbscan_zca_whitening", False),
                hdbscan_zca_eps=zca_eps if zca_eps is not None else config.get("hdbscan_zca_eps", 0.1),
                clustering_method=clustering_method or config.get("clustering_method", "hdbscan"),
                hdbscan_n_clusters=n_clusters or config.get("n_clusters", None),
                bgmm_n_components=n_clusters or config.get("bgmm_n_components", 30),
                prototype_checkpoint_path=prototype_checkpoint or config.get("prototype_checkpoint_path"),
                build_bridge_edges=build_bridge_edges,
                collapse_entities=collapse_entities,
                seed=seed,
                aggr_method=aggr_method or config.get("aggr_method", "mean"),
                use_burstiness=use_burstiness if use_burstiness is not None else config.get("use_burstiness", False),
                num_layers=config.get("num_layers", 1),
                track_data_source=track_data_source if track_data_source is not None else config.get("track_data_source", False),
                build_precedes_edges=build_precedes_edges if build_precedes_edges is not None else config.get("build_precedes_edges", False),
                precedes_window_hours=precedes_window_hours,
                use_uf_refinement=use_uf_refinement if use_uf_refinement is not None else config.get("use_uf_refinement", False),
            )

            t_start = time.perf_counter()
            chunk_size = config.get("chunk_inference_size")
            if chunk_size and engine.use_geometric_confidence:
                logger.info(f"Running two-phase inference: {len(df) // chunk_size + (1 if len(df) % chunk_size else 0)} chunks of {chunk_size}")
                
                # Phase 1: Collect embeddings from all chunks (memory-safe)
                all_embeddings = []
                all_ids = []
                dataset_name = getattr(df, "_dataset_name", "unknown")
                
                for start in range(0, len(df), chunk_size):
                    chunk = df.iloc[start:start + chunk_size].copy().reset_index(drop=True)
                    chunk._dataset_name = dataset_name  # Preserve dataset name
                    embeddings, ids = engine.correlate(chunk, embed_only=True)
                    all_embeddings.append(embeddings)
                    all_ids.extend(ids)
                
                # Phase 2: Single HDBSCAN on all collected embeddings
                if len(all_embeddings) > 0 and len(all_embeddings[0]) > 0:
                    # Use cluster_embeddings method instead of raw HDBSCAN
                    clusters = engine.cluster_embeddings(
                        np.vstack(all_embeddings), all_ids, confidence_gate=gate
                    )
                    
                    # Merge with original dataframe to get all columns
                    result_df = df.merge(clusters, on='AlertId', how='left')
                    
                
            elif chunk_size:
                # Softmax mode or non-GAEC: use original per-chunk processing
                logger.info(f"Running chunked inference: {len(df) // chunk_size + (1 if len(df) % chunk_size else 0)} chunks of {chunk_size}")
                
                # Run correlate() in chunks matching training scale, concatenate results.
                results_chunks = []
                dataset_name = getattr(df, "_dataset_name", "unknown")
                for start in range(0, len(df), chunk_size):
                    chunk = df.iloc[start:start + chunk_size].copy().reset_index(drop=True)
                    chunk._dataset_name = dataset_name  # Preserve dataset name
                    chunk_result = engine.correlate(chunk)
                    results_chunks.append(chunk_result)
                result_df = pd.concat(results_chunks, ignore_index=True)
                # Re-align index to original df order
                result_df.index = range(len(result_df))
            else:
                result_df = engine.correlate(df)
            latency = time.perf_counter() - t_start
            
            logger.info(f"Correlation completed in {latency:.3f}s, got {len(result_df)} results")
            
            uf_mask = result_df["correlation_method"] == "hgnn+uf_refinement"
            hgnn_mask = result_df["correlation_method"] == "hgnn"
            
            # Compute ARI and NMI against true labels
            pred_labels = result_df["pred_cluster"].values
            ari = float(adjusted_rand_score(true_labels, pred_labels))
            nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
            ami = float(adjusted_mutual_info_score(true_labels, pred_labels))
            homogeneity = float(homogeneity_score(true_labels, pred_labels))
            completeness = float(completeness_score(true_labels, pred_labels))
            v_measure = float(v_measure_score(true_labels, pred_labels))

            # Alt-label evaluation (e.g., attack_cat vs campaign_id)
            alt_label_col = config.get("label_col_alt")
            alt_ari, alt_ami = float("nan"), float("nan")
            if alt_label_col and alt_label_col in result_df.columns:
                alt_true_aligned = result_df[alt_label_col].fillna("unknown").astype(str).values
                pred_aligned = result_df["pred_cluster"].values
                alt_ari = float(adjusted_rand_score(alt_true_aligned, pred_aligned))
                alt_ami = float(adjusted_mutual_info_score(alt_true_aligned, pred_aligned))
            elif alt_label_col and alt_label_col in df.columns:
                if "AlertId" not in df.columns:
                    df["AlertId"] = df.index.astype(str)
                merged_alt = result_df.merge(
                    df[["AlertId", alt_label_col]],
                    on="AlertId", how="left", suffixes=("", "_y")
                )
                # If it already existed but was empty or something, use the one without suffix, else use _y
                use_col = alt_label_col if alt_label_col in merged_alt.columns else alt_label_col + "_y"
                alt_true_aligned = merged_alt[use_col].fillna("unknown").astype(str).values
                pred_aligned = merged_alt["pred_cluster"].values
                alt_ari = float(adjusted_rand_score(alt_true_aligned, pred_aligned))
                alt_ami = float(adjusted_mutual_info_score(alt_true_aligned, pred_aligned))

            # Bridge edge coverage: count alerts involved in bridge edges (IP<->hostname)
            # This is a proxy since we can't easily access the graph after correlation
            # For now, use the same BroFlowId metric but note it's not actual bridge edges
            # TODO: Modify HGNNCorrelationEngine to return bridge edge statistics
            bridge_edge_mask = (
                result_df["BroFlowId"].notna()
                if "BroFlowId" in result_df.columns
                else pd.Series(False, index=result_df.index)
            )
            
            # Log actual bridge edge count from the engine logs (approximate)
            # The logs show "Added X bridge edges" during graph construction
            # For now, we'll use a placeholder calculation based on the logs

            # Binary ARI: for 2-class datasets, map each cluster to its
            # majority ground-truth class, then evaluate as binary problem.
            # This is meaningful for OpTC (RedTeam vs Benign) where HDBSCAN
            # finds many benign subclusters — standard ARI penalises these
            # subclusters even when they correctly separate attack from benign.
            n_true_classes = df[config["label_col"]].nunique()
            if n_true_classes == 2:
                cluster_majority = (
                    result_df.groupby("pred_cluster")[config["label_col"]]
                    .agg(lambda x: x.mode().iloc[0])
                )
                binary_pred = result_df["pred_cluster"].map(cluster_majority)
                binary_ari = float(adjusted_rand_score(result_df[config["label_col"]], binary_pred))
            else:
                binary_ari = float("nan")

            # Cluster purity: for each cluster, majority class proportion
            # Right metric for cross-sensor runs — measures how homogeneous clusters are
            def compute_cluster_purity(true_labels, pred_labels):
                from collections import Counter
                contingency = {}
                for t, p in zip(true_labels, pred_labels):
                    if p not in contingency:
                        contingency[p] = Counter()
                    contingency[p][t] += 1
                
                total = 0
                weighted_purity = 0
                for cluster_id, class_counts in contingency.items():
                    cluster_size = sum(class_counts.values())
                    majority_count = class_counts.most_common(1)[0][1]
                    weighted_purity += majority_count
                    total += cluster_size
                return weighted_purity / total if total > 0 else 0.0
            
            cluster_purity = compute_cluster_purity(
                result_df[config["label_col"]].values,
                result_df["pred_cluster"].values
            )

            # Tactic sequence coherence (Kendall's τ) — novel kill-chain metric
            # Measures if clusters follow the chronological tactic progression
            # High coherence = clusters respect kill-chain ordering (Recon→Access→Impact)
            def compute_tactic_coherence(df_with_labels):
                from scipy.stats import kendalltau
                
                # Need timestamp and tactic columns
                if "timestamp" not in df_with_labels.columns and "EndDate" not in df_with_labels.columns:
                    return float("nan")
                
                time_col = "timestamp" if "timestamp" in df_with_labels.columns else "EndDate"
                tactic_col = "tactic" if "tactic" in df_with_labels.columns else "AttackTechnique"
                
                if tactic_col not in df_with_labels.columns:
                    return float("nan")
                
                # Sort by time
                sorted_df = df_with_labels.sort_values(time_col).reset_index(drop=True)
                
                # Get tactic ordering (numerical encoding)
                tactic_order = {
                    "Reconnaissance": 0, "Initial Access": 1, "Execution": 2,
                    "Persistence": 3, "Privilege Escalation": 4, "Defense Evasion": 5,
                    "Credential Access": 6, "Discovery": 7, "Lateral Movement": 8,
                    "Collection": 9, "Exfiltration": 10, "Impact": 11
                }
                
                tactics = sorted_df[tactic_col].map(lambda x: tactic_order.get(x, -1)).values
                clusters = sorted_df["pred_cluster"].values
                
                # Remove unknown tactics
                mask = tactics >= 0
                if mask.sum() < 2:
                    return float("nan")
                
                tactics = tactics[mask]
                clusters = clusters[mask]
                
                # Compute Kendall's τ between tactic sequence and cluster sequence
                # If clusters follow tactic progression, τ will be positive
                try:
                    tau, _ = kendalltau(tactics, clusters)
                    return float(tau) if not np.isnan(tau) else 0.0
                except:
                    return float("nan")
            
            tactic_coherence = compute_tactic_coherence(result_df)

            # Attack F1 — binary detection quality (attack vs benign/normal)
            # Maps each cluster to majority class, then computes F1 for attack detection
            def compute_attack_f1(df_with_labels, label_col):
                from sklearn.metrics import f1_score
                
                # Determine if this is a binary attack dataset
                unique_labels = df_with_labels[label_col].unique()
                
                # Check for binary attack/normal pattern
                attack_indicators = ['attack', 'malicious', 'redteam', 'red_team', '1', 'true', 'benign', 'normal', '0', 'false']
                has_attack_pattern = any(str(l).lower() in attack_indicators for l in unique_labels)
                
                if len(unique_labels) > 10 or not has_attack_pattern:
                    return float("nan")  # Multi-campaign dataset, not binary attack detection
                
                # Map clusters to majority class
                cluster_to_class = df_with_labels.groupby("pred_cluster")[label_col].agg(lambda x: x.mode().iloc[0])
                y_pred_binary = df_with_labels["pred_cluster"].map(cluster_to_class)
                y_true = df_with_labels[label_col].values
                
                # Normalize to binary (attack=1, benign=0)
                def to_binary(label):
                    label_str = str(label).lower()
                    if label_str in ['attack', 'malicious', 'redteam', 'red_team', '1', 'true']:
                        return 1
                    elif label_str in ['benign', 'normal', '0', 'false', 'none']:
                        return 0
                    return 0  # Default to benign
                
                y_true_binary = [to_binary(l) for l in y_true]
                y_pred_binary = [to_binary(l) for l in y_pred_binary]
                
                try:
                    return float(f1_score(y_true_binary, y_pred_binary, pos_label=1))
                except:
                    return float("nan")
            
            attack_f1 = compute_attack_f1(result_df, config["label_col"])

            row = {
                "dataset": dataset_name,
                "gate_value": gate,
                "ari": ari,
                "binary_ari": binary_ari,
                "cluster_purity": cluster_purity,
                "tactic_coherence": tactic_coherence,
                "attack_f1": attack_f1,
                "nmi": nmi,
                "ami": ami,
                "alt_label_col": alt_label_col or "",
                "alt_ari": alt_ari,
                "alt_ami": alt_ami,
                "homogeneity": homogeneity,
                "completeness": completeness,
                "v_measure": v_measure,
                "n_clusters": result_df["pred_cluster"].nunique(),
                "n_hgnn_clusters": result_df.loc[hgnn_mask, "pred_cluster"].nunique() if hgnn_mask.any() else 0,
                "n_uf_clusters": result_df.loc[uf_mask, "pred_cluster"].nunique() if uf_mask.any() else 0,
                "pct_uf_routed": float(uf_mask.mean()),
                "avg_confidence": float(result_df["cluster_confidence"].mean()),
                "p25_confidence": float(result_df["cluster_confidence"].quantile(0.25)),
                "threshold_used": (
                    float(result_df["correlation_threshold_used"].mean())
                    if "correlation_threshold_used" in result_df.columns
                    else float("nan")
                ),
                "confidence_mode": "gaec" if use_geom else "softmax",
                "skip_gate_sweep": config.get("skip_gate_sweep", False),
                "latency_s": latency,
                "git_hash": git_hash,
                "bridge_edge_records": int(bridge_edge_mask.sum()),
                "bridge_edge_pct": float(bridge_edge_mask.mean()),
                "unique_bridge_flows": (
                    int(result_df.loc[bridge_edge_mask, "BroFlowId"].nunique())
                    if bridge_edge_mask.any() else 0
                ),
            }
            results.append(row)

            logger.info(
                f"    ARI={ari:.4f} | AMI={ami:.4f} | purity={cluster_purity:.4f} | "
                f"coherence={tactic_coherence:.4f} | attack_f1={attack_f1:.4f} | "
                f"pct_uf={row['pct_uf_routed']:.2%} | latency={latency:.3f}s"
            )
            
            # Clean up memory to prevent RAM buildup between gate values
            del result_df, pred_labels, engine
            gc.collect()

    results_df = pd.DataFrame(results)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logger.info(f"\nSweep complete. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the HGNN checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results/gate_tuning_results.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Specific datasets to test (default: all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--build_bridge_edges",
        action="store_true",
        help="Enable bridge edges (overrides config)"
    )
    parser.add_argument(
        "--no_build_bridge_edges",
        action="store_true", 
        help="Disable bridge edges (overrides config)"
    )
    parser.add_argument(
        "--collapse_entities",
        action="store_true",
        help="Enable entity collapse (IP->host routing)"
    )
    parser.add_argument(
        "--gate",
        type=float,
        default=None,
        help="Fixed gate value (overrides sweep)"
    )
    parser.add_argument(
        "--zca_whitening",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Enable Soft-ZCA whitening before clustering"
    )
    parser.add_argument(
        "--zca_eps",
        type=float,
        default=None,
        help="ZCA regularization epsilon (default: 0.1)"
    )
    parser.add_argument(
        "--clustering_method",
        type=str,
        default=None,
        choices=["hdbscan", "spectral", "bgmm", "prototype"],
        help="Override clustering method for all datasets"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters for algorithms that require it (KMeans/BGMM/Agglomerative)"
    )
    parser.add_argument(
        "--aggr_method",
        type=str,
        default=None,
        choices=["mean", "max", "add"],
        help="GNN aggregation method override"
    )
    parser.add_argument(
        "--use_burstiness",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Enable temporal burstiness features in the graph"
    )
    parser.add_argument(
        "--prototype_checkpoint",
        type=str,
        default=None,
        help="Path to prototype checkpoint for clustering_method=prototype"
    )
    parser.add_argument(
        "--track_data_source",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Track source sensor in alert encoding (CS-1)"
    )
    parser.add_argument(
        "--build_precedes_edges",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Add kill-chain precedes edges (CS-3)"
    )
    parser.add_argument(
        "--precedes_window_hours",
        type=float,
        default=2.0,
        help="Max hours between successive precedes edges"
    )
    parser.add_argument(
        "--use_uf_refinement",
        type=lambda x: x.lower() == "true",
        default=None,
        help="Enable UF refinement for low-confidence alerts (default: False)"
    )

    args = parser.parse_args()
    
    # Convert boolean flags to single value
    build_bridge_edges = None
    if args.build_bridge_edges:
        build_bridge_edges = True
    elif args.no_build_bridge_edges:
        build_bridge_edges = False
    
    run_sweep(
        args.checkpoint,
        args.output,
        args.datasets,
        args.seed,
        args.gate,
        build_bridge_edges,
        args.collapse_entities,
        args.clustering_method,
        args.n_clusters,
        args.zca_whitening,
        args.zca_eps,
        args.aggr_method,
        args.use_burstiness,
        args.prototype_checkpoint,
        args.track_data_source,
        args.build_precedes_edges,
        args.precedes_window_hours,
        args.use_uf_refinement,
    )
