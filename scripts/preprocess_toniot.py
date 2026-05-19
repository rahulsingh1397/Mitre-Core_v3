"""
TON_IoT Preprocessing Script
==============================

One-time preprocessing to fix TON_IoT dataset issues:
1. No timestamps - add synthetic monotonic timestamps
2. Attack-type sorted rows - shuffle to simulate realistic campaign interleaving
3. Map attack types to MITRE ATT&CK tactics

Usage:
    python scripts/preprocess_toniot.py

Output:
    datasets/TON_IoT/mitre_format.parquet with proper schema
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Attack type → MITRE ATT&CK tactic mapping
TONIOT_TO_MITRE = {
    "scanning": "Reconnaissance",        # T1046 Network Service Scanning
    "password": "CredentialAccess",      # T1110 Brute Force
    "backdoor": "Persistence",           # T1543 Create/Modify System Process
    "injection": "Execution",            # T1059 Command & Scripting Interpreter
    "xss": "Execution",                  # T1059.007 JavaScript
    "mitm": "Collection",                # T1557 Adversary-in-the-Middle
    "ddos": "Impact",                    # T1498 Network Denial of Service
    "dos": "Impact",                     # T1499 Endpoint Denial of Service
    "ransomware": "Impact",              # T1486 Data Encrypted for Impact
    "normal": None,                      # Benign baseline
}

# Campaign ID mapping (integer 0-9)
TYPE_TO_CAMPAIGN = {
    "scanning": 0,
    "password": 1,
    "backdoor": 2,
    "injection": 3,
    "xss": 4,
    "mitm": 5,
    "ddos": 6,
    "dos": 7,
    "ransomware": 8,
    "normal": 9,
}


def preprocess_toniot():
    """Preprocess TON_IoT dataset with shuffling and synthetic timestamps."""
    
    input_path = Path("datasets/TON_IoT/train_test_network.csv")
    output_path = Path("datasets/TON_IoT/mitre_format.parquet")
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    original_count = len(df)
    logger.info(f"Loaded {original_count:,} rows")
    
    # Check for type column
    if "type" not in df.columns:
        logger.error("Required column 'type' not found in CSV")
        return
    
    # Log original type distribution
    type_counts = df["type"].value_counts()
    logger.info("Original type distribution:")
    for t, c in type_counts.items():
        logger.info(f"  {t}: {c:,}")
    
    # Stratified shuffle: preserve class balance but break type-sorted order
    logger.info("Performing stratified shuffle...")
    
    # Group by type, shuffle each group, then interleave
    shuffled_dfs = []
    for attack_type in df["type"].unique():
        subset = df[df["type"] == attack_type].copy()
        subset = subset.sample(frac=1, random_state=42).reset_index(drop=True)
        shuffled_dfs.append(subset)
    
    # Interleave rows from each type to create mixed windows
    # This simulates realistic SOC view where different attack types appear interleaved
    min_len = min(len(s) for s in shuffled_dfs)
    interleaved = []
    
    for i in range(min_len):
        for subset in shuffled_dfs:
            interleaved.append(subset.iloc[i])
    
    # Add remaining rows from larger groups
    for subset in shuffled_dfs:
        if len(subset) > min_len:
            interleaved.extend([subset.iloc[i] for i in range(min_len, len(subset))])
    
    df = pd.DataFrame(interleaved).reset_index(drop=True)
    logger.info(f"Shuffled: {len(df):,} rows")
    
    # Assign synthetic timestamps
    base_date = datetime(2023, 1, 1, 0, 0, 0)
    df["timestamp"] = pd.date_range(
        start=base_date,
        periods=len(df),
        freq="1s"
    )
    logger.info(f"Timestamp range: {df['timestamp'].min()} → {df['timestamp'].max()}")
    
    # Map attack type to MITRE tactic
    df["tactic"] = df["type"].map(lambda x: TONIOT_TO_MITRE.get(x, x))
    
    # Assign campaign_id from type
    df["campaign_id"] = df["type"].map(lambda x: TYPE_TO_CAMPAIGN.get(x, 99))
    
    # Binary label (0=normal, 1=attack)
    df["label"] = (df["type"] != "normal").astype(int)
    
    # Map column names to standard schema
    column_mapping = {
        "src_ip": "src_ip",
        "dst_ip": "dst_ip",
        "src_port": "src_port",
        "dst_port": "dst_port",
        "proto": "protocol",
    }
    
    # Apply mapping where columns exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df[new_col] = df[old_col]
    
    # Add alert_type from original type
    df["alert_type"] = df["type"]
    
    # Select final columns for mitre_format.parquet
    output_columns = [
        "timestamp",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
        "protocol",
        "tactic",
        "campaign_id",
        "label",
        "alert_type",
    ]
    
    # Only include columns that exist
    final_columns = [c for c in output_columns if c in df.columns]
    df_output = df[final_columns].copy()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write parquet
    df_output.to_parquet(output_path, index=False)
    logger.info(f"Written {len(df_output):,} rows to {output_path}")
    
    # Log summary
    logger.info("\nTactic distribution:")
    tactic_counts = df_output["tactic"].value_counts()
    for t, c in tactic_counts.items():
        logger.info(f"  {t}: {c:,}")
    
    logger.info(f"\nTimestamp range: {df_output['timestamp'].min()} → {df_output['timestamp'].max()}")
    logger.info(f"Duration: {df_output['timestamp'].max() - df_output['timestamp'].min()}")
    logger.info(f"Unique IPs: {df_output['src_ip'].nunique()} src, {df_output['dst_ip'].nunique()} dst")
    
    return df_output


if __name__ == "__main__":
    preprocess_toniot()
