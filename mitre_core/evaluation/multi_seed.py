from __future__ import annotations

import pandas as pd


METRIC_COLUMNS = [
    "ami",
    "ari",
    "binary_ari",
    "purity",
    "silhouette_cosine",
    "attack_f1",
    "cluster_attribution_f1",
    "latency_seconds_per_10k",
    "peak_gpu_gb",
]


def aggregate_multi_seed(results: pd.DataFrame) -> pd.DataFrame:
    available = [column for column in METRIC_COLUMNS if column in results.columns]
    summary = results.groupby(["dataset", "method"])[available].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([part for part in column if part]).strip("_") for column in summary.columns]
    return summary
