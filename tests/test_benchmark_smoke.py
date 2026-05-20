from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from benchmark.run_benchmark import main
from mitre_core.evaluation.benchmark import run_benchmark
from mitre_core.evaluation.manifest import sha256_file, write_run_manifest


def test_benchmark_smoke(tmp_path, monkeypatch) -> None:
    output = tmp_path / "benchmark.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_benchmark.py",
            "--methods",
            "benchmark/methods.yaml",
            "--datasets",
            "benchmark/datasets.yaml",
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    assert output.exists()
    df = pd.read_csv(output)
    assert not df.empty
    assert {"dataset", "method", "seed", "ami", "ari"}.issubset(df.columns)
    manifest_path = output.with_name(output.stem + "_manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert Path(manifest["results_path"]).name == output.name
    assert set(df["method"].unique()) == {
        "K-Means (raw)",
        "DBSCAN (raw)",
        "HDBSCAN (raw)",
        "K-Means (emb)",
        "Spectral (emb)",
        "HDBSCAN (emb)",
        "Spectral (raw)",
        "PCA + K-Means",
        "PCA + HDBSCAN",
        "MITRE-CORE V3",
    }


def test_real_dataset_manifest_includes_reproducibility_artifacts(tmp_path) -> None:
    dataset_path = tmp_path / "nsl_kdd_like.csv"
    checkpoint_path = tmp_path / "network_it_best.pt"
    methods_path = tmp_path / "methods.yaml"
    datasets_path = tmp_path / "datasets.yaml"
    output_path = tmp_path / "benchmark_real.csv"
    split_dir = tmp_path / "splits"
    manifest_path = tmp_path / "benchmark_real_manifest.json"

    rows = 24
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min").astype(str),
            "src_ip": [f"10.0.0.{index % 6}" for index in range(rows)],
            "dst_ip": [f"192.168.0.{index % 4}" for index in range(rows)],
            "protocol": ["tcp" if index % 2 == 0 else "udp" for index in range(rows)],
            "service": ["http" if index % 3 == 0 else "smtp" for index in range(rows)],
            "src_bytes": np.arange(rows),
            "dst_bytes": np.arange(rows) * 3,
            "tactic": ["Reconnaissance"] * 8 + ["Execution"] * 8 + ["None"] * 8,
        }
    )
    df.to_csv(dataset_path, index=False)
    checkpoint_path.write_text("mock checkpoint", encoding="utf-8")
    methods_path.write_text(
        yaml.safe_dump(
            {
                "methods": [
                    {
                        "name": "K-Means (raw)",
                        "type": "baseline",
                        "entry": "kmeans_raw",
                        "enabled": True,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    datasets_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "name": "NSL-KDD-dev",
                        "kind": "mitre_format",
                        "path": str(dataset_path),
                        "split_group": "nsl_kdd",
                        "label_col": "tactic",
                        "checkpoint": str(checkpoint_path),
                        "seeds": [101, 202],
                        "sample_size": 12,
                        "sample_seed": 7,
                        "stratified_sample": True,
                        "n_clusters": 3,
                        "split_dir": str(split_dir),
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    results = run_benchmark(
        methods_path=methods_path,
        datasets_path=datasets_path,
        output_path=output_path,
        command=["python", "-m", "benchmark.run_benchmark", "--datasets", str(datasets_path)],
    )

    assert output_path.exists()
    assert len(results) == 2
    manifest = results.attrs["run_manifest"]
    dataset_manifest = manifest["datasets"][0]
    split_path = Path(dataset_manifest["split_indices_path"])

    assert manifest["command_line"] == "python -m benchmark.run_benchmark --datasets " + str(datasets_path)
    assert dataset_manifest["source_path"] == str(dataset_path.resolve())
    assert dataset_manifest["source_sha256"] == sha256_file(dataset_path)
    assert dataset_manifest["checkpoint_path"] == str(checkpoint_path.resolve())
    assert dataset_manifest["checkpoint_sha256"] == sha256_file(checkpoint_path)
    assert dataset_manifest["benchmark_seeds"] == [101, 202]
    assert dataset_manifest["sample_seed"] == 7
    assert dataset_manifest["selected_row_count"] == 12
    assert dataset_manifest["split_group"] == "nsl_kdd"
    assert split_path.exists()
    assert dataset_manifest["split_indices_sha256"] == sha256_file(split_path)

    write_run_manifest(manifest, manifest_path)
    persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert persisted_manifest == manifest


def test_multi_track_label_evaluation(tmp_path) -> None:
    dataset_path = tmp_path / "multi_track.csv"
    methods_path = tmp_path / "methods.yaml"
    datasets_path = tmp_path / "datasets.yaml"
    output_path = tmp_path / "benchmark_multi.csv"
    split_dir = tmp_path / "splits"

    rows = 24
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="min").astype(str),
            "src_ip": [f"10.0.0.{index % 6}" for index in range(rows)],
            "dst_ip": [f"192.168.0.{index % 4}" for index in range(rows)],
            "protocol": ["tcp" if index % 2 == 0 else "udp" for index in range(rows)],
            "service": ["http" if index % 3 == 0 else "smtp" for index in range(rows)],
            "src_bytes": np.arange(rows),
            "dst_bytes": np.arange(rows) * 3,
            "tactic": ["Reconnaissance"] * 8 + ["Execution"] * 8 + ["None"] * 8,
            "alert_type": ["attack"] * 16 + ["normal"] * 8,
            "campaign_id": ["c1"] * 12 + ["c2"] * 8 + ["c3"] * 4,
        }
    )
    df.to_csv(dataset_path, index=False)
    methods_path.write_text(
        yaml.safe_dump(
            {
                "methods": [
                    {
                        "name": "K-Means (raw)",
                        "type": "baseline",
                        "entry": "kmeans_raw",
                        "enabled": True,
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    datasets_path.write_text(
        yaml.safe_dump(
            {
                "datasets": [
                    {
                        "name": "MultiTrack",
                        "kind": "mitre_format",
                        "path": str(dataset_path),
                        "split_group": "multi_track",
                        "label_col": "tactic",
                        "alt_label_cols": ["alert_type", "campaign_id"],
                        "seeds": [101],
                        "sample_size": 12,
                        "sample_seed": 7,
                        "stratified_sample": True,
                        "n_clusters": 3,
                        "split_dir": str(split_dir),
                    }
                ]
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    results = run_benchmark(
        methods_path=methods_path,
        datasets_path=datasets_path,
        output_path=output_path,
        command=["python", "-m", "benchmark.run_benchmark", "--datasets", str(datasets_path)],
    )

    assert output_path.exists()
    assert "label_track" in results.columns
    tracks = set(results["label_track"].unique())
    assert tracks == {"label_col", "alert_type", "campaign_id"}
    # One method, one seed, three tracks = 3 rows
    assert len(results) == 3


def test_clustering_sweep_caches_embeddings_and_selects_winner(tmp_path) -> None:
    from benchmark.clustering_sweep_standalone import (
        _cache_dir,
        _cache_key,
        _load_cached_embeddings,
        _save_cached_embeddings,
        _select_winner,
        _cluster_cached_embeddings,
    )

    cache_dir = tmp_path / "cache"
    key = _cache_key("test-ds", "cp_sha", "split_sha")
    embeddings = np.random.randn(24, 8).astype(np.float32)
    meta = {"dataset_name": "test-ds", "checkpoint_sha256": "cp_sha"}
    _save_cached_embeddings(cache_dir, key, embeddings, meta)
    loaded = _load_cached_embeddings(cache_dir, key)
    assert loaded is not None
    assert np.allclose(loaded[0], embeddings)
    assert loaded[1]["dataset_name"] == "test-ds"

    # Verify grid clustering runs and winner selection works
    labels = np.array(["A"] * 8 + ["B"] * 8 + ["None"] * 8)
    config = {
        "clustering_method": "hdbscan",
        "hdbscan_min_cluster_size": 5,
        "hdbscan_pca_components": 4,
        "hdbscan_cluster_selection_epsilon": 0.0,
    }
    pred = _cluster_cached_embeddings(embeddings, config)
    assert len(pred) == 24

    # Build a tiny results DataFrame for winner selection
    import pandas as pd
    results = pd.DataFrame(
        [
            {"primary_ari": 0.1, "primary_ami": 0.2, "n_clusters": 2},
            {"primary_ari": 0.5, "primary_ami": 0.3, "n_clusters": 3},
            {"primary_ari": 0.5, "primary_ami": 0.4, "n_clusters": 3},
        ]
    )
    winner = _select_winner(results)
    # Should pick the row with highest ARI then highest AMI
    assert winner["primary_ari"] == 0.5
    assert winner["primary_ami"] == 0.4
