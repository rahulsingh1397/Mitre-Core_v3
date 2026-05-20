"""
benchmark/clustering_sweep_standalone.py
-----------------------------------------
WARNING: Standalone HDBSCAN sweeps on cached embeddings do NOT transfer to the
full V3 pipeline. NSL-KDD Phase 3 demonstrated catastrophic ARI degradation
(0.632 → 0.078) when a sweep winner was ported to V3CorrelationEngine. Root
cause: EmbeddingConfidenceScorer post-processing changes the effective embedding
geometry in ways a standalone HDBSCAN call does not replicate.

Use benchmark/clustering_sweep_full_engine.py for sweeps whose winners
will actually be used. This file is retained only as historical reference.

Original docstring:
Clustering hyperparameter sweep on cached HGNN embeddings for a frozen-split
dev subset.  Extracts embeddings once, then re-clusters with a controlled HDBSCAN
grid.  Winner is selected on the primary label track and locked into the
benchmark config.

Usage:
    python -m benchmark.clustering_sweep \
        --dataset-config benchmark/datasets_real.yaml \
        --dataset-name NSL-KDD-dev \
        --output benchmark/results/nsl_kdd_clustering_sweep.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

from mitre_core.evaluation.benchmark import (
    _detect_benign_label,
    _load_dataset,
)
from mitre_core.evaluation.unsupervised_metrics import compute_unsupervised_metrics
from mitre_core.inference.correlation_engine import V3CorrelationEngine

logger = logging.getLogger("clustering_sweep")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(dataset_name: str, checkpoint_sha256: str, split_sha256: str) -> str:
    payload = f"{dataset_name}:{checkpoint_sha256}:{split_sha256}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _cache_dir() -> Path:
    return Path("benchmark/cache/embeddings")


def _load_cached_embeddings(cache_dir: Path, cache_key: str) -> tuple[np.ndarray, dict[str, Any]] | None:
    emb_path = cache_dir / f"{cache_key}.npy"
    meta_path = cache_dir / f"{cache_key}.json"
    if not emb_path.exists() or not meta_path.exists():
        return None
    embeddings = np.load(emb_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return embeddings, metadata


def _save_cached_embeddings(
    cache_dir: Path,
    cache_key: str,
    embeddings: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{cache_key}.npy", embeddings)
    (cache_dir / f"{cache_key}.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Sweep grid
# ---------------------------------------------------------------------------

def _make_grid() -> list[dict[str, Any]]:
    """Controlled HDBSCAN grid for NSL-KDD."""
    configs = []
    for min_cluster_size in [3, 5, 10]:
        for pca_components in [8, 16, 24]:
            for cluster_selection_epsilon in [0.0, 0.05, 0.1, 0.15]:
                configs.append(
                    {
                        "clustering_method": "hdbscan",
                        "hdbscan_min_cluster_size": min_cluster_size,
                        "hdbscan_pca_components": pca_components,
                        "hdbscan_cluster_selection_epsilon": cluster_selection_epsilon,
                    }
                )
    return configs


# ---------------------------------------------------------------------------
# Single-config evaluation on cached embeddings
# ---------------------------------------------------------------------------

def _cluster_cached_embeddings(
    embeddings: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Apply PCA (if configured) and HDBSCAN to cached embeddings."""
    pca_components = config["hdbscan_pca_components"]
    min_cluster_size = config["hdbscan_min_cluster_size"]
    epsilon = config["hdbscan_cluster_selection_epsilon"]

    z = embeddings
    if pca_components is not None and pca_components > 0 and pca_components < embeddings.shape[1]:
        pca = PCA(n_components=pca_components, whiten=True, random_state=42)
        z = pca.fit_transform(embeddings)

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=epsilon,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(z)

    # HDBSCAN returns -1 for noise; remap to 0 for consistency with benchmark
    noise_mask = labels == -1
    if noise_mask.any():
        unique_labels = set(labels[~noise_mask])
        if len(unique_labels) > 0:
            labels[noise_mask] = 0
        else:
            labels[noise_mask] = 0
    return labels.astype(int)


def _evaluate_config(
    embeddings: np.ndarray,
    labels_primary: np.ndarray,
    benign_label_primary: str | int,
    label_tracks: dict[str, np.ndarray] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    start = time.perf_counter()
    pred = _cluster_cached_embeddings(embeddings, config)
    elapsed = time.perf_counter() - start

    n_clusters = int(len(np.unique(pred)))

    # Primary track
    metrics_primary = compute_unsupervised_metrics(
        labels_primary,
        pred,
        embeddings=embeddings,
        benign_label=benign_label_primary,
    )

    row: dict[str, Any] = {
        "clustering_method": config["clustering_method"],
        "hdbscan_min_cluster_size": config["hdbscan_min_cluster_size"],
        "hdbscan_pca_components": config["hdbscan_pca_components"],
        "hdbscan_cluster_selection_epsilon": config["hdbscan_cluster_selection_epsilon"],
        "n_clusters": n_clusters,
        "latency_seconds": elapsed,
        **{f"primary_{k}": v for k, v in metrics_primary.items()},
    }

    # Alternate tracks
    if label_tracks:
        for track_name, track_labels in label_tracks.items():
            track_benign = _detect_benign_label(track_labels)
            track_metrics = compute_unsupervised_metrics(
                track_labels,
                pred,
                embeddings=embeddings,
                benign_label=track_benign,
            )
            for metric_name, value in track_metrics.items():
                row[f"{track_name}_{metric_name}"] = value

    return row


# ---------------------------------------------------------------------------
# Winner selection
# ---------------------------------------------------------------------------

def _select_winner(results: pd.DataFrame) -> pd.Series:
    """Select best config on primary track ARI, with AMI tie-breaker."""
    # Filter out pathological configs (1 cluster or >100 clusters)
    valid = results[(results["n_clusters"] > 1) & (results["n_clusters"] <= 100)]
    if valid.empty:
        valid = results

    # Sort by primary_ari descending, then primary_ami descending
    best_idx = valid.sort_values(
        by=["primary_ari", "primary_ami"],
        ascending=[False, False],
    ).index[0]
    return valid.loc[best_idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_sweep(
    datasets_path: Path,
    dataset_name: str,
    output_path: Path,
    checkpoint_path: Path | None = None,
) -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    datasets = yaml.safe_load(datasets_path.read_text(encoding="utf-8"))["datasets"]
    dataset = next((d for d in datasets if d["name"] == dataset_name), None)
    if dataset is None:
        raise ValueError(f"Dataset '{dataset_name}' not found in {datasets_path}")

    # Use seed 42 for dev split extraction (the dev split seed)
    seed = 42
    loaded = _load_dataset(dataset, seed=seed)
    df = loaded.frame
    labels_primary = loaded.labels
    benign_label_primary = loaded.benign_label
    label_tracks = loaded.label_tracks

    # Resolve checkpoint
    cp = checkpoint_path or Path(dataset.get("checkpoint", ""))
    if not cp or not cp.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cp}")

    # Build cache key
    from mitre_core.evaluation.manifest import sha256_file
    cp_hash = sha256_file(cp) or "unknown"
    split_path_str = loaded.metadata.get("split_indices_path")
    split_hash = sha256_file(split_path_str) or "unknown"
    cache_key = _cache_key(dataset_name, cp_hash, split_hash)
    cache_dir = _cache_dir()

    cached = _load_cached_embeddings(cache_dir, cache_key)
    if cached is not None:
        embeddings, meta = cached
        logger.info(f"Loaded cached embeddings from {cache_dir / cache_key}.npy")
        # Verify metadata matches
        if meta.get("dataset_name") != dataset_name or meta.get("checkpoint_sha256") != cp_hash:
            logger.warning("Cache metadata mismatch. Re-extracting embeddings.")
            cached = None

    if cached is None:
        logger.info("Extracting HGNN embeddings (one-time)...")
        engine = V3CorrelationEngine(
            model_path=str(cp),
            device="cpu",
            pure_unsupervised=True,
            clustering_method="hdbscan",
            hdbscan_min_cluster_size=5,
            hdbscan_pca_components=16,
            hdbscan_cluster_selection_epsilon=0.1,
            use_geometric_confidence=True,
        )
        embeddings = engine.extract_embeddings(df)
        meta = {
            "dataset_name": dataset_name,
            "checkpoint_path": str(cp.resolve()),
            "checkpoint_sha256": cp_hash,
            "split_indices_path": split_path_str,
            "split_indices_sha256": split_hash,
            "n_alerts": int(len(df)),
            "embedding_dim": int(embeddings.shape[1]),
            "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        _save_cached_embeddings(cache_dir, cache_key, embeddings, meta)
        logger.info(f"Cached embeddings to {cache_dir / cache_key}.npy")

    # Run sweep
    grid = _make_grid()
    logger.info(f"Running sweep: {len(grid)} configs on {len(embeddings)} embeddings")
    rows = []
    for i, config in enumerate(grid, start=1):
        row = _evaluate_config(
            embeddings,
            labels_primary,
            benign_label_primary,
            label_tracks,
            config,
        )
        row["config_id"] = i
        rows.append(row)
        logger.info(
            f"  [{i}/{len(grid)}] min_size={config['hdbscan_min_cluster_size']} "
            f"pca={config['hdbscan_pca_components']} eps={config['hdbscan_cluster_selection_epsilon']} "
            f"→ ARI={row['primary_ari']:.4f} AMI={row['primary_ami']:.4f} clusters={row['n_clusters']}"
        )

    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    logger.info(f"Sweep results saved to {output_path}")

    winner = _select_winner(results)
    logger.info(
        f"\nWinner: min_size={winner['hdbscan_min_cluster_size']} "
        f"pca={winner['hdbscan_pca_components']} eps={winner['hdbscan_cluster_selection_epsilon']} "
        f"→ primary_ari={winner['primary_ari']:.4f} primary_ami={winner['primary_ami']:.4f} "
        f"clusters={winner['n_clusters']}"
    )

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Clustering sweep on cached NSL-KDD embeddings")
    parser.add_argument("--dataset-config", default="benchmark/datasets_real.yaml")
    parser.add_argument("--dataset-name", default="NSL-KDD-dev")
    parser.add_argument("--output", default="benchmark/results/nsl_kdd_clustering_sweep.csv")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    args = parser.parse_args()

    run_sweep(
        datasets_path=Path(args.dataset_config),
        dataset_name=args.dataset_name,
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
