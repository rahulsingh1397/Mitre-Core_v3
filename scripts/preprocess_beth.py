#!/usr/bin/env python3
"""
BETH Dataset Preprocessing for MITRE-CORE
Processes IBM BETH dataset (system events + DNS) into mitre_format.parquet

Input files:
- labelled_training_data.csv + labelled_validation_data.csv (system events)
- labelled_2021may-ip-10-100-1-*-dns.csv + ubuntu-dns.csv (DNS events)

Output:
- datasets/BETH/mitre_format.parquet with campaign_id mapping:
  evil=1 → campaign_id=1 (malicious)
  evil=0 → campaign_id=0 (benign)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_iso_timestamp(ts_str):
    """Parse ISO timestamp to float seconds"""
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.timestamp()
    except:
        return 0.0

def preprocess_system_events(csv_path):
    """Process system events CSV to mitre_format rows"""
    logger.info(f"Processing system events: {csv_path}")
    
    # Read CSV in chunks to handle large files
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=50000):
        # Filter to essential columns
        chunk = chunk[['timestamp', 'processName', 'hostName', 'userId', 'eventName', 'sus', 'evil']].copy()
        
        # Convert to mitre_format
        rows = []
        for _, row in chunk.iterrows():
            # Skip invalid rows
            if pd.isna(row['hostName']):
                continue
                
            # IP mapping from hostname (IP embedded in hostname)
            hostname = str(row['hostName'])
            ip_address = ""
            if hostname.startswith('ip-10-100-1-'):
                # Extract IP from hostname like "ip-10-100-1-4" → "10.100.1.4"
                parts = hostname.split('-')
                if len(parts) >= 4:
                    ip_address = f"10.100.1.{parts[3]}"
            
            # Determine campaign_id using sus OR evil
            campaign_id = 1 if (row.get('evil', 0) == 1 or row.get('sus', 0) == 1) else 0
            
            mitre_row = {
                'SourceHostName': hostname,
                'SourceAddress': ip_address,  # Enables bridge edges!
                'SourceUserName': str(int(row['userId'])) if pd.notna(row['userId']) else '',
                'ProcessName': str(row['processName']),
                'Timestamp': float(row['timestamp']),
                'alert_type': str(row['eventName']),  # syscall name
                'campaign_id': campaign_id,
                'DestinationAddress': None
            }
            rows.append(mitre_row)
        
        if rows:
            chunks.append(pd.DataFrame(rows))
    
    if chunks:
        return pd.concat(chunks, ignore_index=True)
    return pd.DataFrame()

def preprocess_dns_events(csv_path):
    """Process DNS events CSV to mitre_format rows"""
    logger.info(f"Processing DNS events: {csv_path}")
    
    df = pd.read_csv(csv_path)
    rows = []
    
    for _, row in df.iterrows():
        # Skip invalid rows
        if pd.isna(row['SourceIP']) or pd.isna(row['SensorId']):
            continue
            
        mitre_row = {
            'SourceHostName': str(row['SensorId']),  # IP↔host mapping
            'SourceAddress': str(row['SourceIP']),    # Explicit IP for bridge
            'SourceUserName': '',
            'ProcessName': str(row['DnsQuery']),      # DNS query as process name
            'Timestamp': parse_iso_timestamp(row['Timestamp']),
            'alert_type': 'DNS_Query',
            'campaign_id': 1 if (row["evil"] or row["sus"]) else 0,  # sus=1 OR evil=1 = attack
            'DestinationAddress': str(row['DestinationIP']) if pd.notna(row['DestinationIP']) else None
        }
        rows.append(mitre_row)
    
    return pd.DataFrame(rows)

def main():
    """Main preprocessing pipeline"""
    base_dir = Path("datasets/BETH")
    output_path = base_dir / "mitre_format.parquet"
    
    logger.info("Starting BETH dataset preprocessing...")
    
    # Process system events
    system_files = [
        base_dir / "labelled_training_data.csv",
        base_dir / "labelled_validation_data.csv"
    ]
    
    system_dfs = []
    for file_path in system_files:
        if file_path.exists():
            df = preprocess_system_events(file_path)
            if not df.empty:
                system_dfs.append(df)
                logger.info(f"System events from {file_path.name}: {len(df)} rows")
    
    # Process DNS events
    dns_files = list(base_dir.glob("labelled_2021may-*-dns.csv"))
    dns_files.extend([base_dir / "labelled_2021may-ubuntu-dns.csv"])
    
    dns_dfs = []
    for file_path in dns_files:
        if file_path.exists():
            df = preprocess_dns_events(file_path)
            if not df.empty:
                dns_dfs.append(df)
                logger.info(f"DNS events from {file_path.name}: {len(df)} rows")
    
    # Combine all data
    all_dfs = system_dfs + dns_dfs
    if not all_dfs:
        logger.error("No data processed!")
        return
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined_df)} rows")
    
    # Quality checks
    campaign_dist = combined_df['campaign_id'].value_counts().sort_index()
    logger.info(f"Campaign distribution:\n{campaign_dist}")
    
    # Check bridge edge potential (rows with both IP and hostname)
    bridge_rows = combined_df[
        (combined_df['SourceHostName'].str.contains('ip-10-100-1-', na=False)) &
        (combined_df['SourceAddress'].str.contains('10.100.1.', na=False))
    ]
    logger.info(f"Bridge edge candidates: {len(bridge_rows)} rows")
    
    # Save to parquet
    combined_df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    
    # Sample verification
    sample = combined_df.sample(5, random_state=42)
    logger.info("Sample rows:\n" + sample.to_string())

if __name__ == "__main__":
    main()
