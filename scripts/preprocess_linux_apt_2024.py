#!/usr/bin/env python3
"""
Linux APT 2024 Preprocessing Script
===================================

Preprocess the Linux-APT-Dataset-2024 to mitre_core_format.csv:
1. Read Processed Version.xlsx (125,898 rows)
2. Map Wazuh alert fields to MITRE-CORE format
3. Convert timestamps and extract graph features

Usage:
    python scripts/preprocess_linux_apt_2024.py

Output:
    datasets/Linux_APT/linux_apt_2024_mitre_format.csv with columns:
    SourceHostName, EndDate, MalwareIntelAttackType, tactic, technique, mitre_id, label
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Process Linux APT 2024 dataset to MITRE format."""
    root = Path("datasets/Linux_APT/Linux-APT-Dataset-2024")
    # Try CSV first (converted from Excel), fallback to xlsx
    csv_file = Path("datasets/Linux_APT/linux_apt_2024_raw.csv")
    xlsx_file = root / "Processed Version.xlsx"
    output_file = Path("datasets/Linux_APT/linux_apt_2024_mitre_format.csv")
    
    # Read from CSV if available (no openpyxl needed), else xlsx
    if csv_file.exists():
        logger.info(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file)
    elif xlsx_file.exists():
        logger.info(f"Reading {xlsx_file}...")
        df = pd.read_excel(xlsx_file)
    else:
        logger.error(f"No input file found. Tried: {csv_file}, {xlsx_file}")
        return
    
    logger.info(f"Loaded {len(df):,} rows")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Column mapping: raw -> MITRE-CORE format
    # Note: CSV columns have escaped backslashes from Excel conversion
    column_mapping = {
        "agent\\.name": "SourceHostName",           # 5 unique hosts
        "timestamp": "EndDate",                    # "Oct 5, 2023 @ 20:21:46.060"
        "rule\\.description": "MalwareIntelAttackType",  # alert rule name
        "rule\\.mitre\\.tactic": "tactic",
        "rule\\.mitre\\.technique": "technique",
        "rule\\.mitre\\.id": "mitre_id",
        "Malicious / General": "label",            # 0=benign, 1=malicious
    }
    
    # Rename available columns
    available_mapping = {k: v for k, v in column_mapping.items() if k in df.columns}
    df_renamed = df.rename(columns=available_mapping).copy()
    
    # Ensure required columns exist
    required_cols = ["SourceHostName", "EndDate", "MalwareIntelAttackType", 
                     "tactic", "technique", "mitre_id", "label"]
    
    for col in required_cols:
        if col not in df_renamed.columns:
            logger.warning(f"Missing column '{col}', creating with defaults")
            if col == "EndDate":
                df_renamed[col] = pd.Timestamp.now()
            elif col == "label":
                df_renamed[col] = 0  # Default to benign
            else:
                df_renamed[col] = "unknown"
    
    # Convert timestamp
    if "EndDate" in df_renamed.columns:
        df_renamed["EndDate"] = pd.to_datetime(
            df_renamed["EndDate"], 
            format="mixed", 
            errors="coerce"
        )
        # Fill NaT with a default timestamp
        df_renamed["EndDate"] = df_renamed["EndDate"].fillna(pd.Timestamp.now())
    
    # Ensure label is integer (0=benign, 1=malicious)
    if "label" in df_renamed.columns:
        df_renamed["label"] = pd.to_numeric(df_renamed["label"], errors="coerce").fillna(0).astype(int)
    
    # Select output columns
    output_cols = ["SourceHostName", "EndDate", "MalwareIntelAttackType", 
                   "tactic", "technique", "mitre_id", "label"]
    df_output = df_renamed[output_cols].copy()
    
    # Save to CSV
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_file, index=False)
    
    # Log statistics
    logger.info("\n" + "="*60)
    logger.info("Linux APT 2024 Preprocessing Complete!")
    logger.info("="*60)
    logger.info(f"Total records: {len(df_output):,}")
    logger.info(f"Malicious: {df_output['label'].sum():,}")
    logger.info(f"Benign: {(df_output['label'] == 0).sum():,}")
    logger.info(f"Unique hosts: {df_output['SourceHostName'].nunique()}")
    logger.info(f"Unique tactics: {df_output['tactic'].nunique()}")
    logger.info(f"Tactic distribution:")
    for tactic, count in df_output['tactic'].value_counts().head(10).items():
        logger.info(f"  - {tactic}: {count:,}")
    logger.info(f"Output: {output_file}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
