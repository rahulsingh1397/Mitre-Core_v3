#!/usr/bin/env python3
"""
CICAPT-IIoT Preprocessing Script
================================

One-time preprocessing to create CICAPT-IIoT mitre_format.parquet:
1. Read phase1_NetworkData.csv and phase2_NetworkData.csv (9.2 GB total)
2. Process in chunks to handle large file size
3. Map attack labels to MITRE ATT&CK tactics
4. Extract temporal features and assign campaign IDs

Usage:
    python scripts/preprocess_cicapt.py

Output:
    datasets/CICAPT-IIoT-Dataset/mitre_format.parquet with columns:
    timestamp, src_ip, dst_ip, src_port, dst_port, protocol,
    tactic, campaign_id, alert_type, label
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CICAPT-IIoT subLabelCat → MITRE ATT&CK tactic mapping
# Based on 58 attack types mapped to MITRE tactics
CICAPT_TO_MITRE = {
    "Normal": None,
    "PortScan": "Reconnaissance",                    # T1046 Network Service Scanning
    "DDoS": "Impact",                                # T1498 Network Denial of Service
    "DoS": "Impact",                                 # T1499 Endpoint Denial of Service
    "BruteForce": "CredentialAccess",                # T1110 Brute Force
    "SQLInjection": "Execution",                   # T1190 Exploit Public-Facing App
    "XSS": "Execution",                              # T1059.007 JavaScript
    "Backdoor": "Persistence",                       # T1505.003 Web Shell
    "Exploits": "InitialAccess",                     # T1190 Exploit Public-Facing App
    "MITM": "LateralMovement",                       # T1557 Adversary-in-the-Middle
    "Botnet": "CommandAndControl",                   # T1071 Application Layer Protocol
    "Keylogger": "Collection",                       # T1056.001 Keylogging
    "DataExfiltration": "Exfiltration",              # T1041 Exfiltration Over C2
    "Ransomware": "Impact",                          # T1486 Data Encrypted for Impact
    "Phishing": "InitialAccess",                     # T1566 Phishing
    "VulnerabilityScan": "Reconnaissance",             # T1046 Network Service Scanning
    "Fingerprinting": "Reconnaissance",              # T1046 Network Service Scanning
    "Spoofing": "DefenseEvasion",                    # T1556 Modify Authentication Process
}

# Attack type → campaign ID mapping (group related attacks)
LABEL_TO_CAMPAIGN = {
    "Normal": 0,
    "PortScan": 1, "VulnerabilityScan": 1, "Fingerprinting": 1,  # Recon
    "DDoS": 2, "DoS": 2,  # DoS
    "BruteForce": 3,  # Credential Access
    "SQLInjection": 4, "XSS": 4, "Exploits": 4,  # Web attacks
    "Backdoor": 5, "Botnet": 5,  # Persistence/C2
    "MITM": 6,  # Lateral movement
    "Keylogger": 7,  # Collection
    "DataExfiltration": 8,  # Exfiltration
    "Ransomware": 9,  # Impact
    "Phishing": 10,  # Initial access
    "Spoofing": 11,  # Defense evasion
}

def load_and_process_chunk(file_path, chunk_size=50000):
    """Process CSV in chunks to handle large file size."""
    columns = [
        'ts', 'flow_duration', 'Header_Length', 'Source IP', 'Destination IP',
        'Source Port', 'Destination Port', 'Protocol Type', 'Protocol_name',
        'label', 'subLabel', 'subLabelCat'
    ]
    
    chunk_iter = pd.read_csv(
        file_path,
        chunksize=chunk_size,
        usecols=columns if all(col in pd.read_csv(file_path, nrows=0).columns 
                              for col in columns) else None
    )
    
    for chunk_num, chunk in enumerate(chunk_iter):
        logger.info(f"Processing chunk {chunk_num + 1} ({len(chunk)} rows)")
        
        # Map column names to standard MITRE format
        column_mapping = {
            'ts': 'timestamp',
            'Source IP': 'src_ip',
            'Destination IP': 'dst_ip',
            'Source Port': 'src_port',
            'Destination Port': 'dst_port',
            'Protocol Type': 'protocol',
            'subLabelCat': 'label'
        }
        
        # Rename columns (handle missing columns gracefully)
        available_cols = {k: v for k, v in column_mapping.items() if k in chunk.columns}
        df = chunk.rename(columns=available_cols).copy()
        
        # Convert timestamp (ts is Unix timestamp)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Fill missing IP columns
        for col in ['src_ip', 'dst_ip']:
            if col not in df.columns:
                df[col] = '0.0.0.0'
        
        # Fill missing port columns
        for col in ['src_port', 'dst_port']:
            if col not in df.columns:
                df[col] = 0
            df[col] = df[col].fillna(0).astype(int)
        
        # Fill missing protocol
        if 'protocol' not in df.columns:
            df['protocol'] = 'TCP'
        df['protocol'] = df['protocol'].fillna('TCP').astype(str)
        
        # Map attack labels
        if 'label' in df.columns:
            label_series = df['label'].squeeze()  # Ensure it's a Series
            if isinstance(label_series, pd.DataFrame):
                label_series = label_series.iloc[:, 0]
            label_series = label_series.fillna('Normal')
            df['label'] = label_series
            df['tactic'] = label_series.map(lambda x: CICAPT_TO_MITRE.get(str(x), 'Unknown'))
            df['campaign_id'] = label_series.map(lambda x: LABEL_TO_CAMPAIGN.get(str(x), 99)).astype(int)
            df['alert_type'] = label_series
        else:
            df['tactic'] = 'Unknown'
            df['campaign_id'] = 0
            df['alert_type'] = 'Normal'
        
        # Select output columns
        output_cols = [
            'timestamp', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
            'tactic', 'campaign_id', 'alert_type', 'label'
        ]
        
        # Ensure all output columns exist
        for col in output_cols:
            if col not in df.columns:
                df[col] = None
        
        yield df[output_cols]

def main():
    """Process both phase files and create mitre_format.parquet."""
    root = Path("datasets/CICAPT-IIoT-Dataset")
    output_path = root / "mitre_format.parquet"
    
    # Process both phases
    all_chunks = []
    
    for phase_file in ["phase1_NetworkData.csv", "phase2_NetworkData.csv"]:
        file_path = root / phase_file
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            continue
        
        logger.info(f"Processing {phase_file}...")
        for chunk in load_and_process_chunk(file_path):
            all_chunks.append(chunk)
            # Log progress periodically
            if len(all_chunks) % 10 == 0:
                logger.info(f"Processed {len(all_chunks)} chunks ({len(all_chunks) * 50000} rows)")
    
    if not all_chunks:
        logger.error("No data processed!")
        return
    
    # Concatenate all chunks
    logger.info(f"Concatenating {len(all_chunks)} chunks...")
    df_combined = pd.concat(all_chunks, ignore_index=True)
    
    # Remove duplicate columns if any
    if df_combined.columns.duplicated().any():
        logger.warning(f"Duplicate columns found: {df_combined.columns[df_combined.columns.duplicated()].tolist()}")
        df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]
    
    # Sort by timestamp for temporal consistency
    logger.info("Sorting by timestamp...")
    df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
    
    # Save to parquet
    logger.info(f"Saving to {output_path}...")
    df_combined.to_parquet(output_path, compression='snappy', index=False)
    
    # Log statistics
    logger.info("\n" + "="*50)
    logger.info("CICAPT-IIoT Preprocessing Complete!")
    logger.info("="*50)
    logger.info(f"Total records: {len(df_combined):,}")
    logger.info(f"Unique tactics: {df_combined['tactic'].nunique()}")
    logger.info(f"Tactic distribution:")
    for tactic, count in df_combined['tactic'].value_counts().items():
        logger.info(f"  - {tactic}: {count:,}")
    logger.info(f"Output: {output_path}")
    logger.info("="*50)

if __name__ == "__main__":
    main()
