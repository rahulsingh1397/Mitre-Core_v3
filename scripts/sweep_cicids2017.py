"""
CICIDS2017-specific full-engine sweep with expanded mcs grid.

Standard sweep grid uses mcs=[3,5,10] designed for smaller class sizes.
CICIDS2017 has BENIGN=7,290 rows in a 10K sample (72.9%); the standard grid
under-samples the mcs space needed to consolidate the BENIGN region.

Expanded grid:
  mcs: [5, 50, 100, 200, 300, 500, 1000]
  pca: [8, 16, 24]
  eps: [0.0, 0.05, 0.1, 0.15]
  sel_method: [eom, leaf]

Total: 7 × 3 × 4 × 2 = 168 configs on dev split (seed=42).

Usage:
    python scripts/sweep_cicids2017.py
"""
from __future__ import annotations

import itertools
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

DATASETS_YAML = Path("benchmark/datasets_real.yaml")
OUTPUT_PATH = Path("benchmark/results/latest/cicids2017/sweep_full_engine.csv")
DEV_DATASET_NAME = "CICIDS2017-dev"

GRID = {
    "hdbscan_min_cluster_size": [5, 50, 100, 200, 300, 500, 1000],
    "hdbscan_pca_components": [8, 16, 24],
    "hdbscan_cluster_selection_epsilon": [0.0, 0.05, 0.1, 0.15],
    "hdbscan_cluster_selection_method": ["eom", "leaf"],
}

DEV_SEED = 42


def main() -> None:
    from mitre_core.evaluation.benchmark import _load_real_dataset
    from mitre_core.evaluation.unsupervised_metrics import compute_unsupervised_metrics
    from mitre_core.inference.correlation_engine import V3CorrelationEngine

    cfg = yaml.safe_load(DATASETS_YAML.read_text())["datasets"]
    ds = next((d for d in cfg if d["name"] == DEV_DATASET_NAME), None)
    if ds is None:
        raise ValueError(f"Dataset '{DEV_DATASET_NAME}' not found in {DATASETS_YAML}")

    # Force enabled for loading
    ds = dict(ds)
    ds["enabled"] = True

    print(f"Loading {DEV_DATASET_NAME} with seed={DEV_SEED}...")
    loaded = _load_real_dataset(ds, DEV_SEED)
    labels = loaded.labels
    print(f"  {len(labels)} rows loaded; {len(set(labels))} classes in dev subset")

    configs = list(itertools.product(
        GRID["hdbscan_min_cluster_size"],
        GRID["hdbscan_pca_components"],
        GRID["hdbscan_cluster_selection_epsilon"],
        GRID["hdbscan_cluster_selection_method"],
    ))
    total = len(configs)
    print(f"Sweep: {total} configs")

    base_engine_kwargs = dict(ds.get("engine_kwargs", {}))
    rows = []

    for i, (mcs, pca, eps, sel_method) in enumerate(configs, 1):
        engine_kwargs = {
            **base_engine_kwargs,
            "hdbscan_min_cluster_size": mcs,
            "hdbscan_pca_components": pca,
            "hdbscan_cluster_selection_epsilon": eps,
            "hdbscan_cluster_selection_method": sel_method,
            "seed": DEV_SEED,
        }
        try:
            t0 = perf_counter()
            engine = V3CorrelationEngine(
                model_path=ds.get("checkpoint"),
                pure_unsupervised=True,
                clustering_method="hdbscan",
                **engine_kwargs,
            )
            output = engine.infer(loaded.frame)
            preds = output.predictions.to_numpy()
            elapsed = perf_counter() - t0
            metrics = compute_unsupervised_metrics(labels, preds)
            rows.append({
                "hdbscan_min_cluster_size": mcs,
                "hdbscan_pca_components": pca,
                "hdbscan_cluster_selection_epsilon": eps,
                "hdbscan_cluster_selection_method": sel_method,
                "ami": metrics["ami"],
                "ari": metrics["ari"],
                "n_pred_clusters": metrics.get("n_pred_clusters", -1),
                "noise_fraction": metrics.get("noise_fraction", 0.0),
                "latency_s": elapsed,
            })
            if i % 20 == 0 or i == total:
                print(f"  [{i}/{total}] mcs={mcs} pca={pca} eps={eps} sel={sel_method} "
                      f"-> ARI={metrics['ari']:.4f} AMI={metrics['ami']:.4f} "
                      f"n_clusters={metrics.get('n_pred_clusters', -1)}")
        except Exception as exc:
            print(f"  FAILED [{i}/{total}] mcs={mcs} pca={pca} eps={eps} sel={sel_method}: {exc}")

    df = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nResults written to {OUTPUT_PATH}")

    if not df.empty:
        summary = df.groupby(
            ["hdbscan_min_cluster_size", "hdbscan_pca_components",
             "hdbscan_cluster_selection_epsilon", "hdbscan_cluster_selection_method"]
        )[["ami", "ari", "n_pred_clusters"]].mean().sort_values("ari", ascending=False)
        print("\n=== TOP 20 CONFIGS BY ARI ===")
        print(summary.head(20).to_string())
        print("\n=== TOP CONFIG ===")
        best = summary.iloc[0]
        print(best)


if __name__ == "__main__":
    main()
