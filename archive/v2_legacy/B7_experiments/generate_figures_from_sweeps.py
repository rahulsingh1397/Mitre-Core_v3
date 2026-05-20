"""
experiments/generate_figures_from_sweeps.py
-------------------------------------------
Figure generator that reads real CSV outputs from 13B/13C/13D.
Six publication-quality figures sourced from actual sweep data.

Usage:
    python experiments/generate_figures_from_sweeps.py
    python experiments/generate_figures_from_sweeps.py --output_dir docs/figures/ablation/
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
ABLATION_DIR = ROOT / "experiments" / "results" / "ablation_studies"
BASELINE_CSV = ROOT / "experiments" / "results" / "baseline_clustering_comparison.csv"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
PALETTE = sns.color_palette("Set2", 8)
UF_COLORS = {"UF Off": "#2ca02c", "UF On": "#d62728"}
DATASET_COLORS = {
    "NSL-KDD": PALETTE[0],
    "UNSW-NB15": PALETTE[1],
    "TON_IoT": PALETTE[2],
    "CICIDS2017": PALETTE[3],
    "SQTK_SIEM_kcluster": PALETTE[4],
    "OpTC": PALETTE[5],
}


def load_uf_ablation():
    """Load 13B UF ablation data, return DataFrame with uf_status column."""
    frames = []
    for ds in ["NSL-KDD", "UNSW-NB15", "TON_IoT"]:
        for uf_label, uf_status in [("uf_off", "UF Off"), ("uf_on", "UF On")]:
            path = ABLATION_DIR / f"uf_ablation_{ds}_{uf_label}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["uf_status"] = uf_status
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_ton_sweep():
    """Load 13D TON_IoT sample size sweep."""
    frames = []
    for sz in [2000, 5000, 10000]:
        path = ABLATION_DIR / f"ton_sample_sweep_{sz}.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["sample_size"] = sz
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_baseline():
    """Load 13C baseline clustering comparison."""
    if BASELINE_CSV.exists():
        return pd.read_csv(BASELINE_CSV)
    return pd.DataFrame()


# ===================================================================
# FIGURE 1: UF Ablation — ARI comparison (UF on vs off)
# ===================================================================
def figure1_uf_ablation_ari(uf_df, out_dir):
    """Bar chart: ARI with UF on vs off across datasets."""
    if uf_df.empty:
        print("Skipping Figure 1: no UF ablation data")
        return

    summary = uf_df.groupby(["dataset", "uf_status"])["ari"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = ["NSL-KDD", "UNSW-NB15", "TON_IoT"]
    x = np.arange(len(datasets))
    width = 0.35

    for i, status in enumerate(["UF Off", "UF On"]):
        vals = []
        for ds in datasets:
            row = summary[(summary["dataset"] == ds) & (summary["uf_status"] == status)]
            vals.append(row["ari"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, vals, width, label=status,
                      color=UF_COLORS[status], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("ARI")
    ax.set_title("Figure 1: Union-Find Refinement Ablation — ARI Impact")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(datasets)
    ax.legend(frameon=True, loc="lower right")
    ax.set_ylim(0, 0.9)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    path = out_dir / "fig1_uf_ablation_ari.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ===================================================================
# FIGURE 2: UF Ablation — Cluster fragmentation
# ===================================================================
def figure2_uf_fragmentation(uf_df, out_dir):
    """Bar chart: n_clusters with UF on vs off."""
    if uf_df.empty:
        print("Skipping Figure 2: no UF ablation data")
        return

    summary = uf_df.groupby(["dataset", "uf_status"])["n_clusters"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = ["NSL-KDD", "UNSW-NB15", "TON_IoT"]
    x = np.arange(len(datasets))
    width = 0.35

    for i, status in enumerate(["UF Off", "UF On"]):
        vals = []
        for ds in datasets:
            row = summary[(summary["dataset"] == ds) & (summary["uf_status"] == status)]
            vals.append(row["n_clusters"].values[0] if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, vals, width, label=status,
                      color=UF_COLORS[status], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{int(val)}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Number of Clusters")
    ax.set_title("Figure 2: UF Refinement Causes Cluster Fragmentation")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(datasets)
    ax.legend(frameon=True, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    path = out_dir / "fig2_uf_fragmentation.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ===================================================================
# FIGURE 3: Baseline Clustering Comparison
# ===================================================================
def figure3_baseline_comparison(baseline_df, out_dir):
    """Grouped bar chart: ARI by method across datasets."""
    if baseline_df.empty:
        print("Skipping Figure 3: no baseline data")
        return

    # Filter to key methods for clarity
    key_methods = ["K-Means", "DBSCAN", "HDBSCAN", "K-Means-emb", "HDBSCAN-emb", "MITRE-CORE"]
    df = baseline_df[baseline_df["method"].isin(key_methods)].copy()

    datasets = sorted(df["dataset"].unique())
    x = np.arange(len(datasets))
    n_methods = len(key_methods)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, method in enumerate(key_methods):
        vals = []
        for ds in datasets:
            row = df[(df["dataset"] == ds) & (df["method"] == method)]
            vals.append(row["ari"].values[0] if len(row) > 0 else 0)
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=method, color=PALETTE[i % len(PALETTE)],
               edgecolor="white", linewidth=0.3)

    ax.set_ylabel("ARI")
    ax.set_title("Figure 3: Baseline Clustering Comparison — All Methods × All Datasets")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.legend(frameon=True, loc="upper right", ncol=2, fontsize=8)
    ax.set_ylim(-0.1, 1.05)
    ax.grid(axis="y", alpha=0.3)
    sns.despine()

    path = out_dir / "fig3_baseline_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ===================================================================
# FIGURE 4: TON_IoT Sample Size Sweep
# ===================================================================
def figure4_ton_sample_sweep(ton_df, out_dir):
    """Line plot: ARI and NMI vs sample size for TON_IoT."""
    if ton_df.empty:
        print("Skipping Figure 4: no TON sweep data")
        return

    summary = ton_df.groupby("sample_size").agg(
        ari_mean=("ari", "mean"), ari_std=("ari", "std"),
        nmi_mean=("nmi", "mean"), n_clusters_mean=("n_clusters", "mean"),
        latency_mean=("latency_s", "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ARI + NMI
    ax1.errorbar(summary["sample_size"], summary["ari_mean"], yerr=summary["ari_std"],
                 marker="o", linewidth=2, markersize=8, label="ARI", color=PALETTE[0], capsize=5)
    ax1.plot(summary["sample_size"], summary["nmi_mean"], marker="s", linewidth=2,
             markersize=8, label="NMI", color=PALETTE[1])
    ax1.set_xlabel("Sample Size")
    ax1.set_ylabel("Score")
    ax1.set_title("Clustering Quality vs Sample Size")
    ax1.legend(frameon=True)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(0, 0.9)

    # Clusters + Latency
    ax2.bar(summary["sample_size"].astype(str), summary["n_clusters_mean"],
            color=PALETTE[2], alpha=0.7, label="n_clusters", edgecolor="white")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(summary)), summary["latency_mean"], marker="D",
                  linewidth=2, markersize=8, color=PALETTE[3], label="latency (s)")
    ax2.set_xlabel("Sample Size")
    ax2.set_ylabel("Number of Clusters", color=PALETTE[2])
    ax2_twin.set_ylabel("Latency (s)", color=PALETTE[3])
    ax2.set_title("Cluster Count & Latency vs Sample Size")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Figure 4: TON_IoT Sample Size Sweep — Why 10K is Canonical", fontsize=13, y=1.02)
    sns.despine()

    path = out_dir / "fig4_ton_sample_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ===================================================================
# FIGURE 5: Gate Sweep Curves (from UF-off data)
# ===================================================================
def figure5_gate_sweep_curves(uf_df, out_dir):
    """Line plot: ARI vs gate value for each dataset."""
    if uf_df.empty:
        print("Skipping Figure 5: no gate sweep data")
        return

    df_off = uf_df[uf_df["uf_status"] == "UF Off"].copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    for ds in sorted(df_off["dataset"].unique()):
        ds_data = df_off[df_off["dataset"] == ds].sort_values("gate_value")
        ax.plot(ds_data["gate_value"], ds_data["ari"], marker="o", linewidth=2,
                markersize=6, label=ds, color=DATASET_COLORS.get(ds, "gray"))

    ax.set_xlabel("Confidence Gate Threshold")
    ax.set_ylabel("ARI")
    ax.set_title("Figure 5: Gate Sensitivity — ARI vs Confidence Threshold")
    ax.legend(frameon=True, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 0.85)
    sns.despine()

    path = out_dir / "fig5_gate_sweep_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ===================================================================
# FIGURE 6: Summary Heatmap — Best method per dataset
# ===================================================================
def figure6_summary_heatmap(baseline_df, out_dir):
    """Heatmap: ARI matrix (datasets × methods) with best highlighted."""
    if baseline_df.empty:
        print("Skipping Figure 6: no baseline data")
        return

    pivot = baseline_df.pivot_table(index="dataset", columns="method", values="ari", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(12, 6))
    mask = pivot.isna()
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1,
                mask=mask, linewidths=0.5, linecolor="white", ax=ax,
                cbar_kws={"label": "ARI"})

    # Highlight best per row
    for i, idx in enumerate(pivot.index):
        best_col = pivot.loc[idx].idxmax()
        j = list(pivot.columns).index(best_col)
        ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="#2ca02c",
                                    linewidth=3, zorder=10))

    ax.set_title("Figure 6: Summary Heatmap — Clustering ARI (Best per dataset outlined)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

    path = out_dir / "fig6_summary_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate figures from real sweep CSVs")
    parser.add_argument("--output_dir", default="docs/figures/ablation",
                        help="Output directory for figures")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    uf_df = load_uf_ablation()
    ton_df = load_ton_sweep()
    baseline_df = load_baseline()

    print(f"  UF ablation: {len(uf_df)} rows")
    print(f"  TON sweep:   {len(ton_df)} rows")
    print(f"  Baseline:    {len(baseline_df)} rows")

    print("\nGenerating figures...")
    figure1_uf_ablation_ari(uf_df, out_dir)
    figure2_uf_fragmentation(uf_df, out_dir)
    figure3_baseline_comparison(baseline_df, out_dir)
    figure4_ton_sample_sweep(ton_df, out_dir)
    figure5_gate_sweep_curves(uf_df, out_dir)
    figure6_summary_heatmap(baseline_df, out_dir)

    print(f"\nDone. {len(list(out_dir.glob('*.png')))} figures saved to {out_dir}")


if __name__ == "__main__":
    main()
