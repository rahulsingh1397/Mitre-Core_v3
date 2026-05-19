"""
Preprocess LANL 2017 Unified Host and Network Dataset (WLS + Netflow day-02)
Converts to mitre_format.parquet for MITRE-CORE HGNN training.

Usage:
    python scripts/preprocess_lanl2017.py

Input:
    - datasets/LANL 2021–2024/HostEvents/wls_day-02/wls_day-02 (JSONL, ~14 GB)
    - datasets/LANL 2021–2024/Netflow/netflow_day-02 (CSV, 6.7 GB)

Output:
    - datasets/lanl2017_day02/mitre_format.parquet

Streaming processing to handle large files without loading into RAM.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

# EventID to alert_type mapping for WLS
EVENT_TYPE_MAP = {
    4688: "Process_Execution",
    4648: "Lateral_Movement",
    4624: "Logon_Success",
    4625: "Logon_Failure",
    4672: "Privilege_Escalation",
    4768: "Kerberos_TGT",
    4769: "Kerberos_Service",
    4776: "NTLM_Auth",
    4634: "Logoff"
}

# Attack-relevant ports for Netflow filtering
ATTACK_PORTS = {445, 3389, 22, 88, 135, 139, 389, 636, 3268, 5985, 5986}
KNOWN_SERVERS = {"EnterpriseAppServer", "ActiveDirectory", "VPN"}

# Batch size for parquet writes
BATCH_SIZE = 1_000_000

# Canonical column order for mitre_format
COLUMN_ORDER = [
    "SourceHostName",
    "DestinationHostName",
    "SourceAddress",
    "DestinationAddress",
    "SourcePort",
    "DestinationPort",
    "Protocol",
    "BytesSent",
    "PacketsSent",
    "Timestamp",
    "alert_type",
    "campaign_id",
    "ProcessName",
    "SourceUserName",
]


def process_wls_jsonl(input_path: Path, writer: pq.ParquetWriter, campaign_counter: List[int]) -> int:
    """
    Process WLS JSONL file line-by-line, writing to parquet in batches.
    
    Returns:
        Number of rows processed
    """
    rows = []
    total_rows = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing WLS", unit="lines"):
            line = line.strip()
            if not line:
                continue
                
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            event_id = event.get("EventID")
            if event_id not in EVENT_TYPE_MAP:
                continue
            
            row = {
                "SourceHostName": event.get("LogHost", ""),
                "SourceUserName": event.get("UserName", ""),
                "ProcessName": event.get("ProcessName", ""),
                "DestinationHostName": event.get("Destination", ""),
                "Timestamp": event.get("Time", 0),
                "alert_type": EVENT_TYPE_MAP.get(event_id, "Other"),
                "campaign_id": 0,  # heuristic until redteam.txt downloaded
                "SourceAddress": "",
                "DestinationAddress": "",
                "SourcePort": 0,
                "DestinationPort": 0,
                "Protocol": 0,
                "BytesSent": 0,
                "PacketsSent": 0,
            }
            
            rows.append(row)
            
            if len(rows) >= BATCH_SIZE:
                df = pd.DataFrame(rows)
                df = df.reindex(columns=COLUMN_ORDER)
                table = pa.Table.from_pandas(df)
                writer.write_table(table)
                total_rows += len(rows)
                rows = []
    
    # Write remaining rows
    if rows:
        df = pd.DataFrame(rows)
        df = df.reindex(columns=COLUMN_ORDER)
        table = pa.Table.from_pandas(df)
        writer.write_table(table)
        total_rows += len(rows)
    
    return total_rows


def should_keep_netflow_row(src: str, dst: str, dst_port: str) -> bool:
    """
    Filter Netflow rows based on attack relevance.
    Keep:
    - All TCP flows to attack ports
    - All flows where src starts with IP* (external→internal)
    - 0.5% random sample of other flows (handled by caller)
    """
    # Check attack ports
    if dst_port.isdigit():
        port = int(dst_port)
        if port in ATTACK_PORTS:
            return True
    
    # Check external→internal
    if src.startswith("IP"):
        return True
    
    return False


def process_netflow_csv(input_path: Path, writer: pq.ParquetWriter) -> int:
    """
    Process Netflow CSV line-by-line, writing to parquet in batches.
    
    Returns:
        Number of rows processed
    """
    rows = []
    total_rows = 0
    random_sample_counter = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing Netflow", unit="lines"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 11:
                continue
            
            time_val, duration, src, dst, protocol, src_port, dst_port, packets, bytes_val, field_9, field_10 = parts[:11]
            
            # Filter: keep attack-relevant flows
            if not should_keep_netflow_row(src, dst, dst_port):
                # Random 0.5% sample for background traffic
                random_sample_counter += 1
                if random_sample_counter % 200 != 0:
                    continue
            
            # Map to mitre_format
            row = {
                "SourceHostName": src if (src.startswith("Comp") or src in KNOWN_SERVERS) else "",
                "DestinationHostName": dst if (dst.startswith("Comp") or dst in KNOWN_SERVERS) else "",
                "SourceAddress": src if src.startswith("IP") else "",
                "DestinationAddress": dst if dst.startswith("IP") else "",
                "SourcePort": int(src_port) if src_port.isdigit() else 0,
                "DestinationPort": int(dst_port) if dst_port.isdigit() else 0,
                "Protocol": int(protocol) if protocol.isdigit() else 0,
                "BytesSent": int(bytes_val) if bytes_val.isdigit() else 0,
                "PacketsSent": int(packets) if packets.isdigit() else 0,
                "Timestamp": int(time_val) if time_val.isdigit() else 0,
                "alert_type": "Network_Flow",
                "campaign_id": 0,
                "ProcessName": "",
                "SourceUserName": "",
            }
            
            rows.append(row)
            
            if len(rows) >= BATCH_SIZE:
                df = pd.DataFrame(rows)
                df = df.reindex(columns=COLUMN_ORDER)
                table = pa.Table.from_pandas(df)
                writer.write_table(table)
                total_rows += len(rows)
                rows = []
    
    # Write remaining rows
    if rows:
        df = pd.DataFrame(rows)
        df = df.reindex(columns=COLUMN_ORDER)
        table = pa.Table.from_pandas(df)
        writer.write_table(table)
        total_rows += len(rows)
    
    return total_rows


def main():
    """Main preprocessing pipeline."""
    print("=" * 70)
    print("LANL 2017 Day-02 Preprocessing: WLS + Netflow → mitre_format")
    print("=" * 70)
    
    # Paths
    base_dir = Path("datasets/LANL 2021–2024")
    wls_path = base_dir / "HostEvents/wls_day-02/wls_day-02"
    netflow_path = base_dir / "Netflow/netflow_day-02"
    output_dir = Path("datasets/lanl2017_day02")
    output_path = output_dir / "mitre_format.parquet"
    
    # Validate inputs
    if not wls_path.exists():
        print(f"ERROR: WLS file not found: {wls_path}")
        print("Download from: https://csr.lanl.gov/data-fence/")
        return 1
    
    if not netflow_path.exists():
        print(f"WARNING: Netflow file not found: {netflow_path}")
        print("Continuing with WLS only...")
        netflow_path = None
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define parquet schema
    schema = pa.schema([
        ("SourceHostName", pa.string()),
        ("DestinationHostName", pa.string()),
        ("SourceAddress", pa.string()),
        ("DestinationAddress", pa.string()),
        ("SourcePort", pa.int64()),
        ("DestinationPort", pa.int64()),
        ("Protocol", pa.int64()),
        ("BytesSent", pa.int64()),
        ("PacketsSent", pa.int64()),
        ("Timestamp", pa.int64()),
        ("alert_type", pa.string()),
        ("campaign_id", pa.int64()),
        ("ProcessName", pa.string()),
        ("SourceUserName", pa.string()),
    ])
    
    # Initialize parquet writer
    with pq.ParquetWriter(output_path, schema) as writer:
        total_wls = 0
        total_netflow = 0
        
        # Process WLS
        if wls_path.exists():
            print(f"\nProcessing WLS: {wls_path}")
            total_wls = process_wls_jsonl(wls_path, writer, [0])
            print(f"WLS rows processed: {total_wls:,}")
        
        # Process Netflow
        if netflow_path and netflow_path.exists():
            print(f"\nProcessing Netflow: {netflow_path}")
            total_netflow = process_netflow_csv(netflow_path, writer)
            print(f"Netflow rows processed: {total_netflow:,}")
    
    # Summary
    total_rows = total_wls + total_netflow
    print("\n" + "=" * 70)
    print("Preprocessing Complete!")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Total rows: {total_rows:,}")
    print(f"  - WLS: {total_wls:,}")
    print(f"  - Netflow: {total_netflow:,}")
    
    # Verify output
    if output_path.exists():
        df_check = pd.read_parquet(output_path)
        print(f"\nVerification:")
        print(f"  - Columns: {list(df_check.columns)}")
        print(f"  - Row count: {len(df_check):,}")
        print(f"  - Alert types: {df_check['alert_type'].value_counts().to_dict()}")
        print(f"  - Unique SourceHostName: {df_check['SourceHostName'].nunique():,}")
        print(f"  - Unique SourceAddress: {df_check['SourceAddress'].nunique():,}")
    
    return 0


if __name__ == "__main__":
    exit(main())
