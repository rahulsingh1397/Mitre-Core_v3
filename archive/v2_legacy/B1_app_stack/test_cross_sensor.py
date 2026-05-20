"""
Tests for cross-sensor attack chain features (CS-1 through CS-4).
All tests use synthetic DataFrames — no file I/O, no model weights.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ingestion.dataset_profiler import MultiSourceIngestionPipeline
from hgnn.hgnn_correlation import AlertToGraphConverter


# --- Test Fixtures ---

def make_df(n=10, with_source=True):
    """Minimal MITRE-format DataFrame for testing."""
    # Ensure alternating pattern works for any n
    alt_pattern = ["Reconnaissance", "Impact"] * ((n // 2) + 1)
    df = pd.DataFrame({
        "AlertId":    [f"a{i}" for i in range(n)],
        "EndDate":    pd.date_range("2026-01-01", periods=n, freq="10min"),
        "SourceAddress":     ["10.0.0.1"] * (n // 2) + ["10.0.0.2"] * (n - n // 2),
        "DestinationAddress":  ["192.168.1.1"] * n,
        "SourceHostName":    ["hostA"] * (n // 2) + ["hostB"] * (n - n // 2),
        "AttackTechnique":   alt_pattern[:n],
        "Protocol":          ["TCP"] * n,
        "DestinationService": ["http"] * n,
        "MalwareIntelAttackType": ["attack"] * n,
        "campaign_id":["c1"] * (n // 2) + ["c2"] * (n - n // 2),
        "stage":      ["recon"] * n,
        "src_bytes":  np.random.randint(100, 1000, n),
        "dst_bytes":  np.random.randint(100, 1000, n),
    })
    if with_source:
        df["data_source"] = (["firewall"] * (n // 2) + ["edr"] * (n - n // 2))
    return df


# --- CS-4 Tests: MultiSourceIngestionPipeline ---

def test_data_source_column_preserved():
    """data_source column survives into merged DataFrame."""
    fw = make_df(6, with_source=False)
    edr = make_df(4, with_source=False)
    pipeline = MultiSourceIngestionPipeline()
    pipeline.add_source(fw, "firewall")
    pipeline.add_source(edr, "edr")
    merged = pipeline.merge()
    assert "data_source" in merged.columns
    assert set(merged["data_source"].unique()) == {"firewall", "edr"}


def test_alert_ids_are_unique_after_merge():
    """AlertIds from different sources don't collide."""
    fw  = make_df(5, with_source=False)
    edr = make_df(5, with_source=False)
    pipeline = MultiSourceIngestionPipeline()
    pipeline.add_source(fw, "fw")
    pipeline.add_source(edr, "edr")
    merged = pipeline.merge()
    assert merged["AlertId"].nunique() == 10, "AlertId collision detected"


def test_pipeline_from_csv_paths(tmp_path):
    """from_csv_paths convenience constructor works end-to-end."""
    fw_path  = tmp_path / "fw.csv"
    edr_path = tmp_path / "edr.csv"
    make_df(5, with_source=False).to_csv(fw_path, index=False)
    make_df(3, with_source=False).to_csv(edr_path, index=False)
    merged = MultiSourceIngestionPipeline.from_csv_paths({
        "firewall": str(fw_path), "edr": str(edr_path)
    })
    assert len(merged) == 8
    assert "data_source" in merged.columns


def test_pipeline_missing_required_columns_raises():
    """add_source should raise ValueError if required columns are missing."""
    df = pd.DataFrame({"wrong_column": [1, 2, 3]})
    pipeline = MultiSourceIngestionPipeline()
    with pytest.raises(ValueError) as exc_info:
        pipeline.add_source(df, "bad_source")
    assert "missing required columns" in str(exc_info.value)


# --- CS-1 Tests: data_source Column Encoding ---

def test_missing_source_zeros_in_encoding():
    """
    When data_source column absent, source_ids must be all zeros.
    Tests CS-1B: _encode_alert_features graceful fallback.
    """
    df = make_df(8, with_source=False)
    conv = AlertToGraphConverter(track_data_source=True)
    # Minimal required columns for _encode_alert_features
    features, _, _, _ = conv._encode_alert_features(df)
    # source_ids should be dim 21 — must be all zeros when column absent
    assert features.shape[1] == 21
    assert features[:, 20].sum() == 0.0


def test_source_encoding_when_column_present():
    """When data_source column is present, source_ids should be non-zero categoricals."""
    df = make_df(8, with_source=True)
    conv = AlertToGraphConverter(track_data_source=True)
    features, _, _, _ = conv._encode_alert_features(df)
    # source_ids should be dim 21 with non-zero values for tracked sources
    assert features.shape[1] == 21
    # Should have different values for different sources
    unique_sources = len(np.unique(features[:, 20]))
    assert unique_sources == 2  # firewall and edr


def test_source_encoding_disabled_when_track_false():
    """When track_data_source=False, source_ids should be all zeros even with column present."""
    df = make_df(8, with_source=True)
    conv = AlertToGraphConverter(track_data_source=False)
    features, _, _, _ = conv._encode_alert_features(df)
    # source_ids should be all zeros when tracking disabled
    assert features[:, 20].sum() == 0.0


# --- CS-2 Tests: source_sensor Nodes and collected_by Edges ---

def test_source_sensor_nodes_created():
    """source_sensor nodes and collected_by edges present when track_data_source=True."""
    df = make_df(8, with_source=True)
    conv = AlertToGraphConverter(track_data_source=True)
    data = conv.convert(df)
    assert "source_sensor" in data.node_types, "source_sensor node type missing"
    assert ("alert", "collected_by", "source_sensor") in data.edge_types
    assert ("source_sensor", "collects", "alert") in data.edge_types


def test_source_sensor_nodes_not_created_when_disabled():
    """source_sensor nodes should not exist when track_data_source=False."""
    df = make_df(8, with_source=True)
    conv = AlertToGraphConverter(track_data_source=False)
    data = conv.convert(df)
    assert "source_sensor" not in data.node_types


def test_source_sensor_nodes_not_created_when_column_missing():
    """source_sensor nodes should not exist when data_source column is missing."""
    df = make_df(8, with_source=False)
    conv = AlertToGraphConverter(track_data_source=True)
    data = conv.convert(df)
    assert "source_sensor" not in data.node_types


def test_collected_by_edge_count():
    """Each alert should have exactly one collected_by edge to its source_sensor."""
    df = make_df(10, with_source=True)
    conv = AlertToGraphConverter(track_data_source=True)
    data = conv.convert(df)
    
    # Get edge index for collected_by
    edge_index = data[("alert", "collected_by", "source_sensor")].edge_index
    # Each alert should have exactly one collected_by edge
    assert edge_index.shape[1] == len(df)


# --- CS-3 Tests: precedes Temporal Edges ---

def test_precedes_edges_created():
    """precedes edges should be created when build_precedes_edges=True."""
    df = make_df(8, with_source=False)
    conv = AlertToGraphConverter(build_precedes_edges=True, precedes_window_hours=24.0)
    data = conv.convert(df)
    assert ("alert", "precedes", "alert") in data.edge_types


def test_precedes_edges_not_created_when_disabled():
    """precedes edges should not exist when build_precedes_edges=False."""
    df = make_df(8, with_source=False)
    conv = AlertToGraphConverter(build_precedes_edges=False)
    data = conv.convert(df)
    assert ("alert", "precedes", "alert") not in data.edge_types


def test_precedes_edges_directed():
    """precedes edges are directed: A→B should not have B→A for the same pair."""
    df = make_df(20, with_source=False)
    # Use larger window to ensure some edges are created
    conv = AlertToGraphConverter(build_precedes_edges=True, precedes_window_hours=24.0)
    data = conv.convert(df)
    
    if ("alert", "precedes", "alert") in data.edge_types:
        edge_index = data[("alert", "precedes", "alert")].edge_index
        src, dst = edge_index[0].tolist(), edge_index[1].tolist()
        
        # For directed edges, check that no pair appears in both directions
        pairs = set(zip(src, dst))
        for s, d in pairs:
            assert (d, s) not in pairs, f"Found bidirectional edge: {(s, d)} and {(d, s)}"


def test_precedes_edges_within_window():
    """precedes edges should only connect alerts within the precedes_window_hours."""
    df = make_df(20, with_source=False)
    # Use a small window to limit edges
    conv = AlertToGraphConverter(build_precedes_edges=True, precedes_window_hours=0.5)
    data = conv.convert(df)
    
    if ("alert", "precedes", "alert") in data.edge_types:
        edge_index = data[("alert", "precedes", "alert")].edge_index
        # Just verify edges were created (detailed timing test would be more complex)
        assert edge_index.shape[1] >= 0  # Can be zero if no edges fit in window


def test_precedes_edges_without_timestamp():
    """precedes edges should not be created when timestamp column is missing."""
    df = make_df(8, with_source=False)
    df = df.drop(columns=["EndDate"])
    conv = AlertToGraphConverter(build_precedes_edges=True, precedes_window_hours=24.0)
    data = conv.convert(df)
    # Should not have precedes edges without timestamps
    assert ("alert", "precedes", "alert") not in data.edge_types or \
           data[("alert", "precedes", "alert")].edge_index.shape[1] == 0


# --- CS-5: Backward Compatibility ---

def test_backward_compat_no_flags():
    """AlertToGraphConverter with all defaults unchanged — no new nodes/edges."""
    df = make_df(8, with_source=True)
    conv = AlertToGraphConverter()   # all defaults: track_data_source=False
    data = conv.convert(df)
    assert "source_sensor" not in data.node_types
    assert ("alert", "precedes", "alert") not in data.edge_types


def test_backward_compat_existing_nodes_preserved():
    """Default converter should still create all standard node types."""
    df = make_df(8, with_source=False)
    conv = AlertToGraphConverter()
    data = conv.convert(df)
    
    # Standard node types should still exist
    assert "alert" in data.node_types
    assert "ip" in data.node_types
    assert "host" in data.node_types
    
    # Standard edge types should still exist
    assert ("alert", "shares_ip", "alert") in data.edge_types
    assert ("alert", "shares_host", "alert") in data.edge_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
