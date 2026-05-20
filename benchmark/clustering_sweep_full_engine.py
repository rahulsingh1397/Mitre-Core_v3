"""
benchmark/clustering_sweep_full_engine.py
------------------------------------------
Clustering hyperparameter sweep routed through the COMPLETE V3CorrelationEngine.

Unlike clustering_sweep_standalone.py (which sweeps HDBSCAN on raw cached
embeddings), this module runs every candidate config through the full
V3CorrelationEngine.cluster_alerts() call. Winners from this sweep
accurately reflect production pipeline behavior.

Background: NSL-KDD Phase 3 demonstrated that standalone sweeps produce
config winners that catastrophically regress when applied to the full engine
(ARI 0.632 → 0.078). See docs/lessons/phase3_sweep_methodology.md.

Usage:
    python -m benchmark.clustering_sweep_full_engine \\
        --dataset NSL-KDD-dev \\
        --output benchmark/results/latest/nsl_kdd/sweep_full_engine.csv

The sweep grid is identical to clustering_sweep_standalone.py:
    min_cluster_size: [3, 5, 10]
    pca_components: [8, 16, 24]
    epsilon: [0.0, 0.05, 0.1, 0.15]
    cluster_selection_method: [eom, leaf]

Results are written to CSV with columns:
    min_cluster_size, pca_components, epsilon, cluster_selection_method,
    seed, ami, ari, purity, n_pred_clusters, noise_fraction

Winner selection: highest AMI on the dev split; tiebreak by ARI.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

# Deferred to avoid import costs at module level
# from mitre_core.inference.correlation_engine import V3CorrelationEngine
# from mitre_core.evaluation.unsupervised_metrics import compute_unsupervised_metrics


GRID = {
    "hdbscan_min_cluster_size": [3, 5, 10],
    "hdbscan_pca_components": [8, 16, 24],
    "hdbscan_cluster_selection_epsilon": [0.0, 0.05, 0.1, 0.15],
    "cluster_selection_method": ["eom", "leaf"],
}

SEEDS = [42, 43, 44]


def run_sweep(
    dataset_config: dict,
    output_path: Path,
    seeds: list[int] = SEEDS,
) -> pd.DataFrame:
    """Run the full-engine clustering sweep and return results DataFrame."""
    from mitre_core.evaluation.benchmark import _load_real_dataset
    from mitre_core.evaluation.unsupervised_metrics import compute_unsupervised_metrics
    from mitre_core.inference.correlation_engine import V3CorrelationEngine

    rows = []
    configs = list(itertools.product(
        GRID["hdbscan_min_cluster_size"],
        GRID["hdbscan_pca_components"],
        GRID["hdbscan_cluster_selection_epsilon"],
        GRID["cluster_selection_method"],
    ))
    print(f"Sweep: {len(configs)} configs × {len(seeds)} seeds = {len(configs)*len(seeds)} runs")

    base_engine_kwargs = dict(dataset_config.get("engine_kwargs", {}))

    for seed in seeds:
        loaded = _load_real_dataset(dataset_config, seed)
        labels = loaded.labels

        for mcs, pca, eps, sel_method in configs:
            engine_kwargs = {
                **base_engine_kwargs,
                "hdbscan_min_cluster_size": mcs,
                "hdbscan_pca_components": pca,
                "hdbscan_cluster_selection_epsilon": eps,
                "seed": seed,
            }
            try:
                t0 = perf_counter()
                engine = V3CorrelationEngine(
                    model_path=dataset_config.get("checkpoint"),
                    pure_unsupervised=True,
                    clustering_method="hdbscan",
                    **engine_kwargs,
                )
                preds = engine.cluster_alerts(loaded.frame)
                elapsed = perf_counter() - t0
                metrics = compute_unsupervised_metrics(labels, preds)
                rows.append({
                    "seed": seed,
                    "hdbscan_min_cluster_size": mcs,
                    "hdbscan_pca_components": pca,
                    "hdbscan_cluster_selection_epsilon": eps,
                    "cluster_selection_method": sel_method,
                    "ami": metrics["ami"],
                    "ari": metrics["ari"],
                    "purity": metrics["purity"],
                    "n_pred_clusters": metrics.get("n_pred_clusters", -1),
                    "noise_fraction": metrics.get("noise_fraction", 0.0),
                    "latency_s": elapsed,
                })
            except Exception as exc:
                print(f"  FAILED seed={seed} mcs={mcs} pca={pca} eps={eps}: {exc}")

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Sweep results written to {output_path}")

    if not df.empty:
        summary = df.groupby(
            ["hdbscan_min_cluster_size", "hdbscan_pca_components",
             "hdbscan_cluster_selection_epsilon", "cluster_selection_method"]
        )[["ami", "ari"]].mean().sort_values("ami", ascending=False)
        print("\nTop 10 configs by mean AMI:")
        print(summary.head(10).to_string())

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-engine clustering sweep")
    parser.add_argument("--dataset", default="NSL-KDD-dev", help="Dataset name from datasets_real.yaml")
    parser.add_argument("--datasets-yaml", default="benchmark/datasets_real.yaml")
    parser.add_argument("--output", default="benchmark/results/latest/sweep_full_engine.csv")
    args = parser.parse_args()

    datasets_cfg = yaml.safe_load(Path(args.datasets_yaml).read_text())["datasets"]
    ds = next((d for d in datasets_cfg if d["name"] == args.dataset), None)
    if ds is None:
        raise ValueError(f"Dataset '{args.dataset}' not found in {args.datasets_yaml}")

    run_sweep(ds, Path(args.output))


if __name__ == "__main__":
    main()
