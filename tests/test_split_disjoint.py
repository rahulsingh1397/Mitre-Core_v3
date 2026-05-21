from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mitre_core.evaluation.benchmark import _load_dataset


def test_dev_eval_splits_are_disjoint_and_persisted(tmp_path) -> None:
    dataset_path = tmp_path / "nsl_kdd_like.csv"
    rows = 60
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min").astype(str),
            "src_ip": [f"10.0.0.{index % 10}" for index in range(rows)],
            "dst_ip": [f"192.168.0.{index % 8}" for index in range(rows)],
            "protocol": ["tcp" if index % 2 == 0 else "udp" for index in range(rows)],
            "service": ["http" if index % 3 == 0 else "smtp" for index in range(rows)],
            "src_bytes": np.arange(rows),
            "dst_bytes": np.arange(rows) * 2,
            "tactic": ["Reconnaissance"] * 20 + ["Execution"] * 20 + ["None"] * 20,
        }
    )
    df.to_csv(dataset_path, index=False)

    split_dir = tmp_path / "splits"
    dev_dataset = {
        "name": "NSL-KDD-dev",
        "kind": "mitre_format",
        "path": str(dataset_path),
        "split_group": "nsl_kdd",
        "label_col": "tactic",
        "sample_size": 15,
        "sample_seed": 42,
        "stratified_sample": True,
        "split_dir": str(split_dir),
    }
    eval_dataset = {
        "name": "NSL-KDD",
        "kind": "mitre_format",
        "path": str(dataset_path),
        "split_group": "nsl_kdd",
        "label_col": "tactic",
        "sample_size": 15,
        "sample_seed": 142,
        "exclude_sample_seeds": [42],
        "stratified_sample": True,
        "split_dir": str(split_dir),
    }

    dev_loaded = _load_dataset(dev_dataset, seed=42)
    eval_loaded = _load_dataset(eval_dataset, seed=42)

    dev_split_path = Path(dev_loaded.metadata["split_indices_path"])
    eval_split_path = Path(eval_loaded.metadata["split_indices_path"])
    assert dev_split_path.exists()
    assert eval_split_path.exists()

    dev_indices = np.load(dev_split_path)
    eval_indices = np.load(eval_split_path)
    assert np.intersect1d(dev_indices, eval_indices).size == 0

    rerun_loaded = _load_dataset(eval_dataset, seed=43)
    rerun_indices = np.load(Path(rerun_loaded.metadata["split_indices_path"]))
    assert np.array_equal(eval_indices, rerun_indices)

    metadata_path = eval_split_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["sample_seed"] == 142
    assert metadata["source_row_count"] == rows


def test_single_split_without_exclusion_covers_full_corpus(tmp_path) -> None:
    """Datasets with no exclude_sample_seeds get a single split covering all rows."""
    dataset_path = tmp_path / "sqtk_siem_like.csv"
    rows = 60
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min").astype(str),
            "src_ip": [f"10.0.0.{index % 10}" for index in range(rows)],
            "dst_ip": [f"192.168.0.{index % 8}" for index in range(rows)],
            "protocol": ["tcp" if index % 2 == 0 else "udp" for index in range(rows)],
            "alert_type": ["blocked"] * 20 + ["alert"] * 20 + ["passed"] * 20,
        }
    )
    df.to_csv(dataset_path, index=False)

    split_dir = tmp_path / "splits"
    dataset = {
        "name": "SQTK_SIEM-like",
        "kind": "mitre_format",
        "path": str(dataset_path),
        "split_group": "sqtk_siem_like",
        "label_col": "alert_type",
        "sample_size": 60,
        "sample_seed": 42,
        "stratified_sample": True,
        "split_dir": str(split_dir),
    }

    loaded = _load_dataset(dataset, seed=42)
    split_path = Path(loaded.metadata["split_indices_path"])
    assert split_path.exists()

    indices = np.load(split_path)
    assert len(indices) == rows, f"Expected {rows} indices, got {len(indices)}"
    assert len(set(indices)) == rows, f"Expected {rows} unique indices, got {len(set(indices))}"

    metadata_path = split_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["sample_seed"] == 42
    assert metadata["source_row_count"] == rows
