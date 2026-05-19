#!/usr/bin/env python3

import sys
sys.path.append('.')
import pandas as pd
from hgnn.hgnn_correlation import HGNNCorrelationEngine
import torch

def debug_bridge_edges():
    """Debug script to check bridge edge construction and usage"""
    
    print("=== DEBUG: Bridge Edge Analysis ===")
    
    # Load a small sample of OpTC data
    df = pd.read_csv("datasets/DARPA_OpTC/processed_optc_full.csv", low_memory=False)
    # Take first 1000 records to avoid sampling issues
    sample_df = df.head(1000).copy()
    
    # Remap columns like the gate tuning script does
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
        if old in sample_df.columns and new not in sample_df.columns:
            sample_df[new] = sample_df[old]
    
    print(f"Sample data shape: {sample_df.shape}")
    print(f"Columns with IP/Host data: {sample_df[['SourceAddress', 'SourceHostName']].notna().sum()}")
    
    # Test with bridge edges ENABLED
    print("\n--- Testing WITH bridge edges ---")
    engine_with = HGNNCorrelationEngine(
        model_path="hgnn_checkpoints/network_v9_v3_bridge/network_it_best.pt",
        device="cpu",
        build_bridge_edges=True
    )
    
    # Get the graph before correlation
    graph_with = engine_with.converter.convert(sample_df)
    print(f"Graph edge types (with bridge): {list(graph_with.edge_index_dict.keys())}")
    print(f"Total edges (with bridge): {sum(edges.shape[1] for edges in graph_with.edge_index_dict.values())}")
    
    # Check if bridge edges exist
    bridge_edges = [et for et in graph_with.edge_index_dict.keys() if '___resolves_to___' in str(et) or '___resolved_from___' in str(et)]
    print(f"Bridge edge types found: {bridge_edges}")
    for et in bridge_edges:
        print(f"  {et}: {graph_with.edge_index_dict[et].shape[1]} edges")
    
    # Test with bridge edges DISABLED  
    print("\n--- Testing WITHOUT bridge edges ---")
    engine_without = HGNNCorrelationEngine(
        model_path="hgnn_checkpoints/network_v9_v3_bridge/network_it_best.pt",
        device="cpu", 
        build_bridge_edges=False
    )
    
    graph_without = engine_without.converter.convert(sample_df)
    print(f"Graph edge types (without bridge): {list(graph_without.edge_index_dict.keys())}")
    print(f"Total edges (without bridge): {sum(edges.shape[1] for edges in graph_without.edge_index_dict.values())}")
    
    # Compare edge counts
    bridge_edge_count = sum(graph_with.edge_index_dict[et].shape[1] for et in bridge_edges)
    print(f"\nBridge edge count difference: {bridge_edge_count}")
    
    # Check model conv dict
    print(f"\n--- Model Architecture Check ---")
    model = engine_with.model
    print(f"Number of conv layers: {len(model.convs)}")
    if hasattr(model.convs[0], 'convs'):
        conv_edge_types = list(model.convs[0].convs.keys())
        print(f"Model conv edge types: {len(conv_edge_types)}")
        bridge_conv_types = [et for et in conv_edge_types if 'resolves' in et or 'resolved' in et]
        print(f"Bridge conv types: {bridge_conv_types}")
    
    print("\n=== DEBUG COMPLETE ===")

if __name__ == "__main__":
    debug_bridge_edges()
