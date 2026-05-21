# CICIDS2017 — Stage 4 Investigation

**Date:** 2026-05-20
**Trigger:** V3 ARI=0.111 vs Spectral(emb) ARI=0.333 — margin = -0.222 (Path B)

---

## Baseline Roster Summary (alert_type / label_col track)

| Method | ARI mean ± std | AMI mean ± std | n_clusters | noise_frac | attack_f1_demoted |
|---|---|---|---|---|---|
| DBSCAN (raw) | 0.395 ± 0.000 | 0.469 ± 0.000 | 2 | 0.908 | **0.000** (degenerate) |
| **Spectral (emb)** | **0.333 ± 0.034** | 0.493 ± 0.032 | 15 | 0.000 | 0.832 |
| PCA + HDBSCAN | 0.171 ± 0.000 | 0.416 ± 0.000 | 208 | 0.083 | 0.953 |
| PCA + K-Means | 0.167 ± 0.023 | 0.520 ± 0.010 | 15 | 0.000 | 0.998 |
| K-Means (emb) | 0.111 ± 0.002 | 0.511 ± 0.001 | 15 | 0.000 | 0.992 |
| **MITRE-CORE V3** | **0.111 ± 0.000** | 0.486 ± 0.000 | **45** | 0.000 | 0.997 |
| Spectral (raw) | 0.104 ± 0.094 | 0.414 ± 0.071 | 15 | 0.000 | 0.775 |
| K-Means (raw) | 0.053 ± 0.014 | 0.392 ± 0.017 | 15 | 0.000 | 0.783 |
| HDBSCAN (raw) | 0.051 ± 0.000 | 0.242 ± 0.000 | 609 | 0.162 | 0.942 |
| HDBSCAN (emb) | 0.026 ± 0.000 | 0.311 ± 0.000 | 276 | 0.014 | 0.998 |

**Degenerate metrics:**
- `dominant_confusion_accuracy`: 1.0 for ALL methods (4th consecutive dataset) — pre-demoted.
- `DBSCAN (raw)`: attack_f1_demoted=0.000 (n_pred_clusters=2 → degenerate binary split).
- V3 is perfectly deterministic: std=0.000 across seeds 42/43/44.

---

## Root Cause Analysis

### Primary cause: BENIGN over-segmentation

V3 produces **45 clusters** (true class count = 15). With HDBSCAN mcs=5:
- BENIGN class: **7,290 rows in the 10K sample** (72.9%), dense, multi-modal
- HDBSCAN with mcs=5 finds fine-grained density modes within BENIGN → ~30 sub-clusters
- Attack classes (12 types): ~2,700 rows → ~15 clusters (roughly correct)
- Total: ~30 + ~15 = ~45 clusters

The result: every named attack type has a roughly correct cluster, but BENIGN is fragmented
into ~30 parts. ARI penalizes this heavily because the 7,290 BENIGN rows are split across many clusters.

### Why Spectral (emb) wins

Spectral benefits from two privileged facts:
1. **n_clusters=15**: receives the exact true class count as a hyperparameter
2. **GNN embeddings**: operates on V3's own embedding space (same embeddings as V3 uses)

Spectral is literally "spectral clustering on V3 embeddings with the exact right cluster count".
V3's disadvantage is that HDBSCAN must discover the cluster count from data, and with massive
class imbalance it defaults to over-segmentation.

### Hypothesis to test

Increasing `hdbscan_min_cluster_size` forces smaller density modes to merge. With mcs in [50, 100,
200, 300, 500, 1000]:
- BENIGN sub-clusters should merge into 1–3 large clusters
- Attack clusters (100–750 rows each) should remain intact until mcs gets very large
- Expected ARI improvement: if BENIGN consolidates to 1 cluster and attack types remain separated,
  ARI could reach 0.3–0.4

### Secondary hypothesis: PCA components and epsilon

- Higher PCA (24) may improve separation in embedding space
- Higher epsilon (0.15) may merge near-clusters in BENIGN
- `cluster_selection_method=leaf` may give finer attack cluster retention

---

## Sweep Design

**Target:** CICIDS2017-dev split (seed=42, 10K rows)
**Grid (expanded for CICIDS2017 class-size regime):**
- mcs: [5, 50, 100, 200, 300, 500, 1000]
- pca: [8, 16, 24]
- eps: [0.0, 0.05, 0.1, 0.15]
- sel_method: [eom, leaf]

Total: 7 × 3 × 4 × 2 = 168 configs × 1 dev seed = 168 runs (single seed for speed)

**Winner selection:** highest ARI on dev split (primary); AMI tiebreak.

**Transfer test:** if winner beats Spectral(emb) 0.333 ARI on dev, evaluate on eval split
(3 seeds) for final frozen numbers.

---

## Sweep Results

**Script:** `scripts/sweep_cicids2017.py`
**Output:** `benchmark/results/latest/cicids2017/sweep_full_engine.csv`
**Total configs run:** 168 (7 mcs × 3 pca × 4 eps × 2 sel_method), single seed (dev, seed=42)

### Top configs by ARI (dev subset)

| mcs | pca | eps | sel_method | ARI | AMI | n_clusters |
|---|---|---|---|---|---|---|
| 200 | 8 | 0.15 | eom | **0.177** | 0.570 | 11 |
| 200 | 8 | 0.15 | leaf | 0.177 | 0.570 | 11 |
| 300 | 8 | 0.15 | eom | 0.177 | 0.570 | 11 |
| 300 | 8 | 0.15 | leaf | 0.177 | 0.570 | 11 |
| 500 | 16 | 0.15 | eom | 0.177 | 0.549 | 9 |
| 1000 | 24 | 0.15 | leaf | 0.177 | 0.549 | 9 |
| 5 (default) | 8 | 0.15 | eom | 0.176 | 0.570 | 16 |

**Key finding:** `hdbscan_cluster_selection_epsilon=0.15` is the dominant driver — it merges
near-clusters from over-segmented BENIGN region. The mcs value matters less at high eps.
Default mcs=5 with eps=0.15 performs nearly identically to mcs=200 (0.176 vs 0.177 ARI).

**Sweep winner:** mcs=200, pca=8, eps=0.15, sel_method=eom → ARI=0.177 on dev (11 clusters).

**Comparison to baseline:**
- Default config: ARI=0.111 (45 clusters, mcs=5, eps=0.1, pca=16)
- Sweep winner dev: ARI=0.177 (11 clusters, mcs=200, eps=0.15, pca=8) → +0.066 improvement
- Spectral (emb) target: ARI=0.333 — gap remains at 0.156 after sweep
- Best non-degenerate baseline (Spectral emb) still wins by +0.156 ARI

**Hypothesis validation:**
- ✅ Over-segmentation confirmed (45 → 11 clusters with larger mcs + eps)
- ✅ BENIGN region consolidates with eps=0.15
- ❌ Gap to Spectral(emb) NOT closed — Spectral's privileged n_clusters=15 advantage persists

---

## Decision

**Freeze with sweep winner config** (mcs=200, pca=8, eps=0.15):
- ARI improvement is real (+60% relative, 0.111 → 0.177 dev)
- 11 clusters vs 45 is a geometrically better result
- Unlike TON_IoT where default config was used (gap was smaller), here the default config
  (0.111, 45 clusters) is poorly calibrated for CICIDS2017's extreme class imbalance
- Swap winner config reduces cluster count from 3× over (45) to close to truth (11 vs 15)

**V3 still loses on CICIDS2017** — this is a genuine finding, not a failure of the lifecycle.
Spectral with GNN embeddings + privileged cluster count wins. This is the second consecutive
dataset where V3 loses (after TON_IoT). The zero-shot claim holds for network IDS only.

**Eval run (3 seeds) pending** — `baseline_roster_sweep_winner.csv` in progress.
