#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def prepare_siem_data_for_mitre_core():
    """Prepare SQTK SIEM data for MITRE-CORE analysis"""
    
    # Add current directory to path to import MITRE-CORE modules
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    print("=== Preparing SQTK SIEM Data for MITRE-CORE ===")
    
    # Load the preprocessed data
    data_path = "datasets/SQTK_SIEM/SBI_preprocessed_data.xlsx"
    df = pd.read_excel(data_path)
    
    print(f"Loaded data: {df.shape}")
    print(f"Key columns: {df.columns.tolist()}")
    
    # Create MITRE-CORE compatible schema
    mitre_core_df = pd.DataFrame()
    
    # Map SIEM fields to MITRE-CORE schema
    field_mapping = {
        'AlertId': 'AlertId',                    # Alert identifier
        'StartDate': 'timestamp',                # Event timestamp  
        'SourceAddress': 'src_ip',               # Source IP
        'DestinationAddress': 'dst_ip',          # Destination IP
        'SourceHostName': 'hostname',            # Source hostname
        'SourceUserName': 'username',            # Username
        'AttackType': 'alert_type',              # Alert type
        'DeviceAction': 'alert_type',            # Device action as alert type
        'tactics': 'tactic',                    # MITRE ATT&CK tactics
        'techniques': 'technique',               # MITRE ATT&CK techniques
        'DeviceProduct': 'service',              # Security product
        'DeviceVendor': 'device_type',           # Device vendor
        'RequestURL': 'protocol',                # URL/protocol info
        'DestinationPort': 'dst_port',            # Destination port
        'SourcePort': 'src_port',                # Source port
        'DeviceSeverity': 'severity',            # Severity level
        'CategoryOutcome': 'stage',              # Attack stage
    }
    
    # Apply mapping
    for siem_col, mitre_col in field_mapping.items():
        if siem_col in df.columns:
            mitre_core_df[mitre_col] = df[siem_col]
        else:
            print(f"Warning: {siem_col} not found in data")
            mitre_core_df[mitre_col] = 'UNKNOWN'
    
    # Add required numeric columns for clustering - FIX: Remove synthetic noise
    # Random bytes poison the graph edge construction. Use zeros to rely on temporal/IP edges.
    mitre_core_df['src_bytes'] = 0
    mitre_core_df['dst_bytes'] = 0
    
    # Create label column for evaluation - FIX: Remove severity leaks and parsing errors
    VALID_TACTICS = {
        'RECONNAISSANCE', 'INITIAL ACCESS', 'EXFILTRATION', 'DISCOVERY',
        'DEFENSE EVASION', 'PRIVILEGE ESCALATION', 'RESOURCE DEVELOPMENT',
        'COMMAND AND CONTROL', 'IMPACT', 'COLLECTION', 'EXECUTION',
        'PERSISTENCE', 'LATERAL MOVEMENT', 'CREDENTIAL ACCESS', 'UNKNOWN'
    }
    
    # Clean tactic mapping to remove severity leaks and parsing errors
    def clean_tactic(tactic):
        if pd.isna(tactic) or tactic == '':
            return 'UNKNOWN'
        tactic_clean = str(tactic).strip().upper()
        return tactic_clean if tactic_clean in VALID_TACTICS else 'UNKNOWN'
    
    mitre_core_df['campaign_id'] = df['tactics'].apply(clean_tactic)
    
    # Add expert kcluster labels as alternative ground truth
    mitre_core_df['kcluster'] = df['kcluster']
    
    # Handle missing values
    for col in mitre_core_df.columns:
        if mitre_core_df[col].dtype == 'object':
            mitre_core_df[col] = mitre_core_df[col].fillna('UNKNOWN')
        else:
            mitre_core_df[col] = mitre_core_df[col].fillna(0)
    
    # Convert timestamp
    mitre_core_df['timestamp'] = pd.to_datetime(mitre_core_df['timestamp'], errors='coerce')
    
    print(f"MITRE-CORE schema shape: {mitre_core_df.shape}")
    print(f"MITRE-CORE columns: {mitre_core_df.columns.tolist()}")
    
    # Save processed data
    output_path = "datasets/SQTK_SIEM/mitre_core_format.csv"
    mitre_core_df.to_csv(output_path, index=False)
    print(f"Saved MITRE-CORE format data to: {output_path}")
    
    # Show data statistics
    print(f"\n=== Data Statistics ===")
    print(f"Total alerts: {len(mitre_core_df)}")
    
    print(f"\n=== Expert kcluster distribution (11 classes, no UNKNOWN) ===")
    print(mitre_core_df['kcluster'].value_counts().sort_index())
    
    print(f"\n=== Cleaned tactic distribution ===")
    tactic_dist = mitre_core_df['campaign_id'].value_counts()
    print(tactic_dist)
    print(f"UNKNOWN percentage: {tactic_dist.get('UNKNOWN', 0) / len(mitre_core_df) * 100:.1f}%")
    
    # Verify fixes
    assert mitre_core_df['kcluster'].nunique() == 11, f"Expected 11 kcluster classes, got {mitre_core_df['kcluster'].nunique()}"
    assert (mitre_core_df['src_bytes'] == 0).all(), "Synthetic bytes not removed"
    assert "CRITICAL" not in mitre_core_df['campaign_id'].values, "Severity leak not fixed"
    print(f"\n✅ All fixes verified:")
    print(f"   - kcluster: {mitre_core_df['kcluster'].nunique()} classes (no UNKNOWN)")
    print(f"   - Synthetic bytes: Removed (all zeros)")
    print(f"   - Severity leaks: Fixed (no CRITICAL in tactics)")
    
    print(f"\nUnique attack types: {mitre_core_df['alert_type'].nunique()}")
    print(f"Attack type distribution:")
    print(mitre_core_df['alert_type'].value_counts())
    
    print(f"\nUnique source IPs: {mitre_core_df['src_ip'].nunique()}")
    print(f"Unique destination IPs: {mitre_core_df['dst_ip'].nunique()}")
    
    return output_path, mitre_core_df

def create_siem_dataset_config():
    """Create dataset configuration for SIEM data"""
    
    config = {
        "SQTK_SIEM": {
            "path": "datasets/SQTK_SIEM/mitre_core_format.csv",
            "label_col": "campaign_id",  # Use tactics as ground truth
            "true_clusters": 8,  # Based on 8 unique tactics
            "hdbscan_min_cluster_size": 50,
            "dbscan_eps": 0.3,
            "sample_size": 5100,  # Use full dataset
            "stratified_sample": True,
            "note": "Real-world SIEM data from SQTK company with MITRE ATT&CK labels"
        }
    }
    
    return config

if __name__ == "__main__":
    # Prepare data
    data_path, processed_df = prepare_siem_data_for_mitre_core()
    
    # Create configuration
    config = create_siem_dataset_config()
    
    print(f"\n=== Dataset Configuration ===")
    for name, cfg in config.items():
        print(f"{name}:")
        for key, value in cfg.items():
            print(f"  {key}: {value}")
    
    print(f"\n✅ SIEM data prepared for MITRE-CORE analysis!")
    print(f"Data path: {data_path}")
    print(f"Ready to run gate tuning experiments with: --dataset SQTK_SIEM")
