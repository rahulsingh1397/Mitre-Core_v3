"""
run_multiseed_headline_table.py
================================

Multi-seed re-runner for the §3 headline table in MITRE-design.md.

Addresses the reviewer point that every reported number in §3 is single-seed
(seed=42). Re-runs ``run_gate_tuning.py`` over seeds {42, 43, 44} (default),
collects the per-seed best (AMI, ARI, binary_ARI) per dataset, and emits a
mean ± std summary CSV suitable for replacing the §3 table.

This script does NOT retrain. It only re-runs HGNN inference + clustering with
different seeds (HDBSCAN ``random_state``, NumPy/Torch). Cost: ~1× the existing
single-seed sweep per added seed (~30–60 min per seed on the canonical
checkpoint, depending on dataset selection).

Usage
-----
    python experiments/run_multiseed_headline_table.py \
        --checkpoint hgnn_checkpoints/network_v9_v3/network_it_best.pt \
        --datasets NSL-KDD UNSW-NB15 TON_IoT OpTC SQTK_SIEM CICIDS2017 \
        --seeds 42 43 44 \
        --output experiments/results/multiseed_headline_table.csv

Outputs
-------
- ``<output>``: Long-form CSV with one row per (dataset, seed, gate) — exactly
  the schema produced by ``run_gate_tuning.py``, with an extra ``seed`` column.
- ``<output>.summary.csv``: Aggregated mean ± std per dataset, taking the best
  gate per seed first (best-by-AMI), then mean ± std across seeds.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("multiseed")

DEFAULT_DATASETS = ["NSL-KDD", "UNSW-NB15", "TON_IoT", "OpTC", "SQTK_SIEM_kcluster", "CICIDS2017"]
DEFAULT_SEEDS = [42, 43, 44]


def run_one_seed(checkpoint: str, datasets: list[str], seed: int, output_dir: Path) -> Path:
    """Invoke run_gate_tuning.py for a single seed, returning the resulting CSV path."""
    out_csv = output_dir / f"gate_tuning_seed{seed}.csv"
    if out_csv.exists():
        logger.info(f"[seed={seed}] reusing existing results at {out_csv}")
        return out_csv

    cmd = [
        sys.executable,
        "experiments/run_gate_tuning.py",
        "--checkpoint", checkpoint,
        "--datasets", *datasets,
        "--seed", str(seed),
        "--output", str(out_csv),
    ]
    logger.info(f"[seed={seed}] running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        logger.error(f"[seed={seed}] run_gate_tuning.py exited non-zero ({completed.returncode}). "
                     f"Continuing with remaining seeds; this seed will be excluded from aggregation.")
        return None
    return out_csv


def aggregate(per_seed_csvs: list[Path], primary_metric: str = "ami") -> pd.DataFrame:
    """Aggregate per-seed CSVs to mean ± std per dataset.

    For each (dataset, seed), pick the row that maximises ``primary_metric``
    (default AMI), then compute mean and std across seeds for AMI / ARI /
    binary_ari (when present).
    """
    frames = []
    for csv_path in per_seed_csvs:
        if csv_path is None or not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        # Ensure seed column is present (run_gate_tuning.py already emits it,
        # but be defensive).
        if "seed" not in df.columns:
            df["seed"] = int(csv_path.stem.split("seed")[-1])
        frames.append(df)

    if not frames:
        raise RuntimeError("No seed CSVs found — aggregation aborted.")

    long_df = pd.concat(frames, ignore_index=True)

    if primary_metric not in long_df.columns:
        raise KeyError(f"Primary metric '{primary_metric}' not found in columns: {list(long_df.columns)}")

    # Pick best gate per (dataset, seed) by primary_metric.
    best_per_seed = (
        long_df.sort_values(primary_metric, ascending=False)
               .groupby(["dataset", "seed"], as_index=False)
               .head(1)
    )

    metric_cols = [c for c in ("ami", "ari", "binary_ari", "purity", "n_clusters") if c in best_per_seed.columns]
    summary = (
        best_per_seed.groupby("dataset")[metric_cols]
                     .agg(["mean", "std", "count"])
                     .round(4)
    )
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary = summary.reset_index()
    return long_df, best_per_seed, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--checkpoint", required=True,
                        help="Path to canonical HGNN checkpoint, e.g. "
                             "hgnn_checkpoints/network_v9_v3/network_it_best.pt")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS,
                        help=f"Datasets to evaluate (default: {' '.join(DEFAULT_DATASETS)})")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS,
                        help=f"Seeds to run (default: {DEFAULT_SEEDS})")
    parser.add_argument("--output", type=str,
                        default="experiments/results/multiseed_headline_table.csv",
                        help="Output CSV (long-form). Summary written to <output>.summary.csv")
    parser.add_argument("--primary_metric", default="ami", choices=["ami", "ari", "binary_ari"],
                        help="Metric used to pick best gate per (dataset, seed) before aggregation")
    parser.add_argument("--per_seed_dir", type=str, default="experiments/results/multiseed_per_seed",
                        help="Directory for per-seed CSVs (also used as cache to avoid re-running)")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return 2

    per_seed_dir = Path(args.per_seed_dir)
    per_seed_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = []
    for seed in args.seeds:
        csv_paths.append(run_one_seed(args.checkpoint, args.datasets, seed, per_seed_dir))

    long_df, best_per_seed, summary = aggregate(csv_paths, primary_metric=args.primary_metric)

    out_long = Path(args.output)
    out_long.parent.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(out_long, index=False)
    logger.info(f"Long-form results -> {out_long} ({len(long_df)} rows)")

    out_summary = out_long.with_suffix(".summary.csv")
    summary.to_csv(out_summary, index=False)
    logger.info(f"Mean +/- std summary -> {out_summary}")

    out_best = out_long.with_suffix(".best_per_seed.csv")
    best_per_seed.to_csv(out_best, index=False)
    logger.info(f"Best gate per (dataset, seed) -> {out_best}")

    print("\n=== Multi-seed summary (best gate per seed by", args.primary_metric, ") ===")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
