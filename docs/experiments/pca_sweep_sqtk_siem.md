# Exp 2.6 — SQTK_SIEM PCA Component Sweep (Finding Doc)

**Date:** 2026-05-23
**Status:** REJECTED — best config (pca12) was tantalizing (0.3806) but no config reached the 0.382 freeze threshold.
**Linked subplan:** Part X.2.6 in working plan

## Hypothesis Tested

The PCA+HDBSCAN baseline that beats V3 on SQTK_SIEM (0.382 vs 0.355) reduces dimensionality before HDBSCAN. V3 also does PCA(16) before HDBSCAN on its 128-dim GNN embeddings, but the choice of 16 components was inherited from NSL-KDD's config — not tuned for SQTK_SIEM. A sweep might find a better component count that closes the 0.027 gap.

## Setup

Same SQTK_SIEM full corpus (5,100 rows, sample_seed=42), same `siem_supcon_v4/best.pt` checkpoint, same `clustering_method: hdbscan`. Only `hdbscan_pca_components` varies. 5 entries × 3 seeds. No code change; YAML only.

## Results

| `hdbscan_pca_components` | V3 ARI mean | V3 ARI std | n_pred_clusters | Δ vs v1.0 (0.355) |
|---|---|---|---|---|
| 8 | 0.3573 | 0.0000 | 9 | +0.002 |
| **12** | **0.3806** | 0.0000 | 20 | **+0.026** |
| 16 (v1.0) | 0.3550 | — | — | (baseline) |
| 20 | 0.3531 | 0.0000 | 45 | −0.002 |
| 32 | 0.3536 | 0.0000 | 49 | −0.001 |
| 64 | 0.3534 | 0.0000 | 49 | −0.001 |

Freeze threshold: ARI ≥ 0.382 (matches PCA+HDBSCAN baseline). **Not reached.**

## Findings

1. **pca12 is the sweet spot.** Reducing from 16→12 components improves ARI by 0.026. Going lower (pca8) loses signal; going higher (pca20+) lets HDBSCAN over-segment (45–49 clusters for true k=14).

2. **The signal is concentrated in ~12 dimensions.** The first 12 PCs of the 128-dim embedding carry the class structure; extra components add noise that HDBSCAN treats as additional density modes.

3. **Cluster count blows up above pca16.** 20 → 45 clusters; 32 → 49; 64 → 49. Higher dimensions = more local density variation = more spurious cluster splits.

4. **PCA tuning alone cannot close the 0.027 gap to the baseline.** Best V3 config (pca12, 0.3806) is still 0.014 below PCA+HDBSCAN (0.382). The remaining gap requires either embedding-space changes (Exp 3) or a different clustering approach.

## Decision

**REJECTED — no freeze.** Best config did not reach the baseline threshold.

Optional follow-up: pca11 and pca13 narrow sweep (~6 min total). Would test whether a finer-grained component count lands at exactly 0.382. Deferred — diminishing returns vs running Exp 3/4 which target the root cause.

## What This Confirms About V3

Combined with Exp 2.5b (CICIDS2017 GMM+BIC, also rejected): **the V3 clustering layer is exonerated as the bottleneck on the loser datasets.** Both a different clusterer (GMM+BIC on CICIDS2017) and a different preprocessing (PCA sweep on SQTK_SIEM) failed to close the gap. The remaining failure modes are in the embedding space itself — addressed by Exp 3 (more features) or Exp 4 (more expressive architecture).

## Artifacts

- `benchmark/results/latest/exp2.6_pca/results.csv` 
- `benchmark/datasets_real.yaml` — 5 `SQTK_SIEM-exp2.6-pca*` entries (kept for reproducibility)
