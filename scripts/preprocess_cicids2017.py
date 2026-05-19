"""
CICIDS2017 Preprocessing Script
================================

One-time preprocessing to create CICIDS2017 mitre_format.parquet:
1. Read all 8 CSVs from TrafficLabelling/ folder
2. Concatenate in day order (Monday → Friday) preserving real temporal ordering
3. Parse timestamps (handles mixed formats: MM/DD/YYYY HH:MM:SS and M/D/YYYY H:MM)
4. Sort by timestamp ascending for chronological order across all files
5. Map attack labels to MITRE ATT&CK tactics
6. Assign campaign IDs for temporal grouping

Usage:
    python scripts/preprocess_cicids2017.py

Output:
    datasets/CICIDS2017/mitre_format.parquet with columns:
    timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
    tactic, campaign_id, alert_type, label
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CICIDS2017 Label → MITRE ATT&CK tactic mapping
# Uses actual label bytes from data (em-dash is \x96 in latin-1 encoding)
CICIDS_TO_MITRE = {
    "BENIGN": None,
    "PortScan": "Reconnaissance",                    # T1046 Network Service Scanning
    "FTP-Patator": "CredentialAccess",               # T1110.001 Brute Force: Password Guessing
    "SSH-Patator": "CredentialAccess",               # T1110.004 Brute Force: Password Spraying
    "Web Attack \x96 Brute Force": "CredentialAccess",   # T1110 Brute Force (em-dash from latin-1)
    "Web Attack \x96 XSS": "Execution",                # T1059.007 JavaScript
    "Web Attack \x96 Sql Injection": "Execution",      # T1190 Exploit Public-Facing App
    "Heartbleed": "InitialAccess",                   # T1190 Exploit Public-Facing App
    "Bot": "CommandAndControl",                      # T1071 Application Layer Protocol
    "Infiltration": "LateralMovement",               # T1557 Adversary-in-the-Middle
    "DoS Hulk": "Impact",                            # T1499 Endpoint Denial of Service
    "DoS GoldenEye": "Impact",                       # T1499 Endpoint Denial of Service
    "DoS slowloris": "Impact",                       # T1499 Endpoint Denial of Service
    "DoS Slowhttptest": "Impact",                    # T1499 Endpoint Denial of Service
    "DDoS": "Impact",                                # T1498 Network Denial of Service
}

# Campaign ID mapping (integer 0-14) - uses actual label bytes with em-dash
LABEL_TO_CAMPAIGN = {
    "BENIGN": 0,
    "PortScan": 1,
    "FTP-Patator": 2,
    "SSH-Patator": 3,
    "Web Attack \x96 Brute Force": 4,
    "Web Attack \x96 XSS": 5,
    "Web Attack \x96 Sql Injection": 6,
    "Heartbleed": 7,
    "Bot": 8,
    "Infiltration": 9,
    "DoS Hulk": 10,
    "DoS GoldenEye": 11,
    "DoS slowloris": 12,
    "DoS Slowhttptest": 13,
    "DDoS": 14,
}

# Day order for CICIDS2017 (Monday July 3 → Friday July 7, 2017)
DAY_ORDER = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv",
]


def preprocess_cicids2017():
    """Preprocess CICIDS2017 dataset with proper temporal ordering."""
    
    input_dir = Path("datasets/CICIDS2017/TrafficLabelling/TrafficLabelling")
    output_path = Path("datasets/CICIDS2017/mitre_format.parquet")
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    # Find all CSV files
    all_csv_files = {f.name: f for f in input_dir.glob("*.csv")}
    logger.info(f"Found {len(all_csv_files)} CSV files in {input_dir}")
    for fname in sorted(all_csv_files.keys()):
        logger.info(f"  - {fname}")
    
    # Update DAY_ORDER to match actual file names (handle case differences)
    available_files = set(all_csv_files.keys())
    day_order_adjusted = []
    for expected in DAY_ORDER:
        # Try exact match first, then case-insensitive
        if expected in available_files:
            day_order_adjusted.append(expected)
        else:
            # Look for case-insensitive match
            for actual in available_files:
                if actual.lower() == expected.lower():
                    day_order_adjusted.append(actual)
                    break
    
    # Process files in day order
    dfs = []
    total_rows = 0
    
    for day_file in day_order_adjusted:
        if day_file not in all_csv_files:
            logger.warning(f"Expected file not found: {day_file}")
            continue
        
        file_path = all_csv_files[day_file]
        logger.info(f"Reading {day_file}...")
        
        try:
            # Read CSV with latin-1 encoding to handle special characters
            df = pd.read_csv(file_path, low_memory=False, encoding='latin-1')
            # Strip leading/trailing spaces from column names
            df.columns = df.columns.str.strip()
            df['_source_file'] = day_file
            dfs.append(df)
            total_rows += len(df)
            logger.info(f"  Loaded {len(df):,} rows")
        except Exception as e:
            logger.error(f"Failed to read {day_file}: {e}")
            continue
    
    if not dfs:
        logger.error("No data files could be loaded")
        return
    
    # Concatenate all dataframes
    logger.info(f"\nConcatenating {len(dfs)} files with {total_rows:,} total rows...")
    df = pd.concat(dfs, ignore_index=True)
    
    # Parse timestamps - CICIDS2017 uses DD/MM/YYYY format (dayfirst=True)
    logger.info("Parsing timestamps...")
    
    # Custom parser for CICIDS2017 mixed formats:
    # - '03/07/2017 08:55:58' (DD/MM/YYYY HH:MM:SS with leading zeros)
    # - '4/7/2017 8:54' (D/M/YYYY H:MM without leading zeros)
    from datetime import datetime
    
    def parse_cicids_timestamp(ts_val):
        """Parse CICIDS2017 timestamp with multiple format attempts."""
        if pd.isna(ts_val):
            return pd.NaT
        ts_str = str(ts_val).strip()
        
        # CICIDS2017 uses D/M/YYYY H:MM format (no leading zeros)
        # strptime with %d/%m/%Y handles both with and without leading zeros
        formats = [
            '%d/%m/%Y %H:%M:%S',  # 03/07/2017 08:55:58 or 3/7/2017 8:55:58
            '%d/%m/%Y %H:%M',     # 03/07/2017 08:55 or 3/7/2017 8:55
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(ts_str, fmt)
            except ValueError:
                continue
        
        # Fallback: try pandas with dayfirst for any other format
        try:
            return pd.to_datetime(ts_str, dayfirst=True, errors='raise')
        except:
            return pd.NaT
    
    # Apply parsing
    logger.info("  Parsing with custom format handler...")
    df['timestamp'] = df['Timestamp'].apply(parse_cicids_timestamp)
    
    # Check for parsing failures and log sample of failed formats
    null_ts = df['timestamp'].isna().sum()
    if null_ts > 0:
        logger.warning(f"  {null_ts:,} rows failed timestamp parsing")
        # Show sample of raw timestamp values that failed
        failed_samples = df[df['timestamp'].isna()]['Timestamp'].dropna().unique()[:5]
        for ts in failed_samples:
            logger.warning(f"    Sample failed: {repr(ts)}")
    
    # Sort by timestamp ascending (ensures chronological order)
    logger.info("Sorting by timestamp...")
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
    
    # Verify Label column exists
    if 'Label' not in df.columns:
        logger.error("Required column 'Label' not found")
        # Check for alternative column names
        label_cols = [c for c in df.columns if 'label' in c.lower()]
        if label_cols:
            logger.info(f"Found alternative label columns: {label_cols}")
        return
    
    # Log original label distribution
    logger.info("\nOriginal label distribution:")
    label_counts = df['Label'].value_counts()
    for label, count in label_counts.items():
        # Show raw bytes for debugging
        label_repr = repr(label)
        logger.info(f"  {label_repr}: {count:,}")
    
    # Note: Labels are read with latin-1 encoding where em-dash = \x96
    # No normalization needed - mapping tables use actual byte values
    logger.info("Mapping labels to MITRE tactics...")
    df['tactic'] = df['Label'].map(lambda x: CICIDS_TO_MITRE.get(x, x))
    
    # Map Label → campaign_id
    df['campaign_id'] = df['Label'].map(lambda x: LABEL_TO_CAMPAIGN.get(x, 99))
    
    # Binary label (0=BENIGN, 1=attack)
    df['label'] = (df['Label'] != 'BENIGN').astype(int)
    
    # Rename columns to standard schema (columns already stripped of spaces)
    column_mapping = {
        'Timestamp': 'timestamp',
        'Source IP': 'src_ip',
        'Destination IP': 'dst_ip',
        'Source Port': 'src_port',
        'Destination Port': 'dst_port',
        'Protocol': 'protocol',
        'Label': 'alert_type',
    }
    
    logger.info("Renaming columns to standard schema...")
    for old_col, new_col in column_mapping.items():
        if new_col == 'timestamp':        # Already parsed as datetime at line 148+
            continue
        if old_col in df.columns:
            df[new_col] = df[old_col]
            logger.info(f"  {old_col} → {new_col}")
        else:
            logger.warning(f"  Column not found: {old_col}")
    
    # Ensure protocol is string
    if 'protocol' in df.columns:
        df['protocol'] = df['protocol'].astype(str)
    
    # Select final columns for mitre_format.parquet
    output_columns = [
        'timestamp',
        'src_ip',
        'dst_ip',
        'src_port',
        'dst_port',
        'protocol',
        'tactic',
        'campaign_id',
        'alert_type',
        'label',
    ]
    
    # Only include columns that exist
    final_columns = [c for c in output_columns if c in df.columns]
    df_output = df[final_columns].copy()
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write parquet
    df_output.to_parquet(output_path, index=False)
    logger.info(f"\nWritten {len(df_output):,} rows to {output_path}")
    
    # Log summary
    logger.info("\nTactic distribution:")
    tactic_counts = df_output['tactic'].value_counts()
    for t, c in tactic_counts.items():
        logger.info(f"  {t}: {c:,}")
    
    logger.info("\nCampaign distribution:")
    campaign_counts = df_output['campaign_id'].value_counts().sort_index()
    for cid, c in campaign_counts.items():
        logger.info(f"  Campaign {cid}: {c:,}")
    
    # Handle NaT timestamps in summary
    ts_min = df_output['timestamp'].min()
    ts_max = df_output['timestamp'].max()
    ts_count = df_output['timestamp'].notna().sum()
    ts_null = df_output['timestamp'].isna().sum()
    
    if pd.notna(ts_min) and pd.notna(ts_max):
        logger.info(f"\nTimestamp range: {ts_min} → {ts_max}")
        logger.info(f"Duration: {ts_max - ts_min}")
    else:
        logger.warning(f"\nTimestamp parsing incomplete: {ts_null} null values")
        logger.info(f"Valid timestamps: {ts_count:,} / {len(df_output):,}")
    
    if 'src_ip' in df_output.columns:
        logger.info(f"Unique src IPs: {df_output['src_ip'].nunique()}")
    if 'dst_ip' in df_output.columns:
        logger.info(f"Unique dst IPs: {df_output['dst_ip'].nunique()}")
    
    # Log benign vs attack split
    benign_count = (df_output['label'] == 0).sum()
    attack_count = (df_output['label'] == 1).sum()
    logger.info(f"\nBenign: {benign_count:,} ({100*benign_count/len(df_output):.1f}%)")
    logger.info(f"Attack: {attack_count:,} ({100*attack_count/len(df_output):.1f}%)")
    
    return df_output


if __name__ == "__main__":
    preprocess_cicids2017()
