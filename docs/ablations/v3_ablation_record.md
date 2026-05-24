# V3 Ablation Record

> **Replaces:** `archive/fabricated_figures/ablation_studies/*.csv` (V2 placeholders — fabricated, not from real runs).
>
> **Format:** Every Part X experiment contributes one row per (dataset) tested, regardless of outcome. Failed treatments are scientific evidence and belong here. The machine-readable record is `v3_ablation_record.csv`; this file is the human view.
>
> **Update protocol:** Append-only. Never rewrite history. Each experiment adds rows at the bottom; freezes link back to the row that motivated them.

## Current Standings (post-Exp 2.6b, 2026-05-23)

| Dataset | Best V3 ARI | Baseline beat? | Frozen version | Notes |
|---|---|---|---|---|
| NSL-KDD | 0.602 | +0.188 over Spectral(raw) | v1.0 | Untouched by Part X |
| UNSW-NB15 | 0.564 | +0.210 over PCA+HDBSCAN | v1.0 | Untouched by Part X |
| TON-IoT | 0.604 | −0.018 vs K-Means(raw) | **v1.1** (GMM+BIC) | Margin closed from −0.199 |
| SQTK_SIEM | 0.461 | **+0.079 over PCA+HDBSCAN** | **v1.1** (pca11) | Flipped from losing to winning |
| CICIDS2017 | 0.177 | −0.156 vs Spectral(emb) | v1.0 | **Sole remaining loser**; needs Exp 3 |
| DARPA OpTC | 0.999 binary | Tied | v1.0 | At ceiling |

## Experiments Executed

### Exp 1 — Hard Negative Mining (REJECTED)

Replaced random negative sampling in topological NT-Xent with 75% hard + 25% random. Loss plateaued at epoch 4/150 — classic hard-mining collapse. CICIDS2017 regressed −0.0425; TON-IoT +0.0091 (sub-threshold). Documented as a measurement-discipline lesson: future experiments require a 10-epoch sanity gate before committing to a full 150-epoch run.

| Dataset | Δ ARI | Decision |
|---|---|---|
| CICIDS2017 | −0.0425 | rejected |
| TON-IoT | +0.0091 | rejected (sub-threshold) |

### Exp 2 S1 — Embedding Collapse Diagnostic (HYPOTHESIS REJECTED)

CPU diagnostic (no training). Measured mean pairwise cosine similarity across 5 datasets × 3 layer depths. **All 15 measurements pass** (cosine_sim 0.60–0.79, all below the 0.90 collapse threshold). The v1.0 SQTK_SIEM "embedding collapse" diagnosis (cosine_sim=0.958) was a measurement artifact from a hardcoded `alert_feature_dim=6` bug; actual value is 0.79.

Triggered the X.2.D documentation correction batch (5 SQTK_SIEM docs + 2 new finding docs, all staged with corrections; v1.0 frozen artifacts unchanged).

### Exp 2.5 — GMM+BIC Clustering Swap

| Dataset | Δ ARI | Decision |
|---|---|---|
| TON-IoT | **+0.1814** | ✅ **freeze v1.1** (BIC selected k=20) |
| CICIDS2017 | −0.0951 | rejected (BIC saturated at k_max=30) |

TON-IoT's HDBSCAN k-discovery failure is fixed. CICIDS2017 needed a retry with a tighter k cap (see 2.5b).

### Exp 2.5b — GMM+BIC CICIDS2017 Retry (REJECTED)

Re-ran CICIDS2017 with `gmm_bic_k_max=15` (matches true class count). BIC selected k=14 deterministically — no saturation. **ARI Δ = −0.0033** vs v1.0. This is the cleanest possible refutation of the clusterer-bottleneck hypothesis for CICIDS2017: when both HDBSCAN and a clean-k-14 GMM produce the same ARI, the clusterer is exonerated. The 72.9% BENIGN class lacks discoverable sub-structure in the embedding space itself.

### Exp 2.6 — SQTK_SIEM PCA Component Sweep (TANTALIZING; SUPERSEDED)

Coarse sweep on `hdbscan_pca_components`. Best: pca12 at 0.3806 (+0.026 vs v1.0). Did not reach the 0.382 baseline threshold. Above pca=16: over-segmentation (45–49 clusters for true k=14).

### Exp 2.6b — SQTK_SIEM Narrow PCA Sweep (FREEZE)

Tested pca11 and pca13 (immediate neighbors of pca12).

| Config | V3 ARI | Decision |
|---|---|---|
| pca11 | **0.4608** | ✅ **freeze v1.1** (+0.106 vs v1.0; +0.079 vs best baseline) |
| pca13 | 0.3587 | rejected |

Knife-edge optimum at pca=11 — deterministic across 3 seeds but a single-component spike. Sensitivity documented in `docs/datasets/sqtk_siem/v1.1_baseline.md`.

## Pending Experiments

- **Exp 3 — 15-dim features:** retrain with `alert_feature_dim=15`. Primary target CICIDS2017. Will add 6 rows (one per dataset).
- **Exp 4 — HGT architecture:** retrain with PyG HGTConv. Most likely to push winners (NSL-KDD, UNSW). Will add 6 rows.
- **Exp 5 — Heterogeneous PE:** conditional on Exp 4 result.

## How to Append a Row

When an experiment runs:

1. For each dataset benchmarked, append one row to `v3_ablation_record.csv`
2. Update this markdown view's "Current Standings" if any dataset's best-known ARI changed
3. Add an "Exp N — Short Name (DECISION)" section to "Experiments Executed"
4. If the experiment produced a freeze, ensure `frozen_path` and `frozen_version` columns are populated and the `decision` column is `freeze_v1.x`

---

## 2026-05-24 Audit — Validity Annotations

The v1.0 Integrity Audit (Stage B materiality test) measured V3 ARI when the leaked
input columns (`tactics`, `alert_types`) are shuffled at inference. Three new
CSV columns capture the result:

- `clean_treatment_ari` — V3 ARI with leaked inputs shuffled
- `delta_vs_clean_baseline` — `clean_treatment_ari` minus the v1.0 clean baseline
   for the same dataset (when the row is a treatment experiment); blank otherwise
- `clean_verdict` — `improvement_genuine` / `regression` / `not_measured` / `not_applicable`

**Two rows have a definitive clean verdict:**

| Experiment | Dataset | Verdict |
|---|---|---|
| 2.5 (GMM+BIC) | TON-IoT v1.1 | **improvement_genuine** — Δ_clean = +0.243 |
| 2.6b (pca=11) | SQTK_SIEM v1.1 | **regression** — Δ_clean = −0.026 |

**TON-IoT v1.1 is the strongest validated result in the program.** SQTK_SIEM v1.1
remains a frozen artifact (immutable) but is marked INVALIDATED in MASTER_PLAN_v1.2
and must not be propagated as a recommendation.

**Other rows** are marked `not_measured` because the materiality test was scoped
to v1.0 baselines + v1.1 winning configs. Intermediate sweep configs (e.g.
SQTK_SIEM pca=8/12/20/32/64, CICIDS2017 hard-neg, TON-IoT hard-neg) were not
re-run with shuffled features. They can be filled in if a future audit pass re-runs
those configs; for now their `clean_*` cells stay empty.

The audit verdict for each (dataset, version) measured baseline is consolidated
in the validity ledger in `docs/plans/MASTER_PLAN_v1.2.md`. The per-experiment
view (this record) and the per-dataset view (master plan ledger) point at the
same underlying measurements.
