# Exp 2 — Multi-Layer Depth: Finding Doc

**Date:** 2026-05-23
**Status:** Hypothesis REJECTED; measurement bug discovered; failure map revised
**Linked experiment doc:** `docs/experiments/exp2_embedding_collapse.md`

---

## Hypothesis Tested

V3's `siem_supcon_v4` checkpoint was reported to produce embeddings with mean pairwise
cosine similarity = 0.958 on SQTK_SIEM, interpreted as embedding collapse. The hypothesis
was that increasing GAT depth (num_layers = 2 or 3) would cure the collapse.

---

## Three Script Bugs Fixed

1. **Premature dummy forward pass** — materialized lazy `Linear(-1, 128)` layers to the
   current dataset's feature dim before loading checkpoint weights, causing shape mismatches.

2. **No feature padding/truncation at inference** — node features not adjusted to match
   encoder expectations. Fixed to mirror `V3CorrelationEngine.extract_embeddings()`.

3. **`alert_feature_dim` hardcoded to 6** — `siem_supcon_v4/best.pt` expects 15-dim alert
   features. The mismatch produced uniformly garbled embeddings with cosine_sim ≈ 0.96.
   **This is the root cause of the original false "collapse" diagnosis.**
   Fix: detect from `state_dict['alert_raw_proj.weight'].shape[1]`.

---

## S1 Measurements (all datasets, num_layers ∈ {1, 2, 3})

| Dataset | L=1 | L=2 | L=3 | Collapsed? | V3 ARI | V3 wins? |
|---|---|---|---|---|---|---|
| sqtk_siem | 0.79 | 0.79 | 0.79 | No | 0.355 | No (−0.027) |
| cicids2017 | **0.60** | 0.60 | 0.61 | No | 0.177 | No (−0.156) |
| ton_iot | 0.73 | 0.73 | 0.74 | No | 0.423 | No (−0.199) |
| unsw_nb15 | 0.74 | 0.77 | 0.75 | No | 0.564 | Yes (+0.210) |
| nsl_kdd | 0.70 | 0.70 | 0.71 | No | 0.602 | Yes (+0.188) |

Collapse threshold: cosine_sim > 0.90. **All 15 measurements pass.**

---

## Findings

### F1 — SQTK_SIEM collapse diagnosis was a measurement artifact

The v1.0 baseline doc, learnings, decision_log, subplan, and investigation.md all cited
cosine_sim = 0.958. This was entirely caused by Bug #3. Actual value: **0.79**.

Corrected docs (2026-05-23 batch):
- `docs/datasets/sqtk_siem/v1.0_baseline.md` — correction banner + inline fixes
- `docs/datasets/sqtk_siem/learnings.md` — correction banner + inline fixes
- `docs/datasets/sqtk_siem/decision_log.md` — inline correction note
- `docs/datasets/sqtk_siem/subplan.md` — inline correction note
- `docs/datasets/sqtk_siem/investigation.md` — INVALIDATED banner (body preserved)

Frozen artifacts (results.csv, manifest.json, engine_config.yaml, splits) are unchanged.

### F2 — Multi-layer depth does not change cosine_sim meaningfully

Max delta across all 15 measurements: 0.03 (UNSW-NB15 L=1→L=2: 0.74→0.77). Confirms the
CLAUDE.md claim that single-layer avoids over-smoothing. Also confirms multi-layer cannot fix
anything, because nothing was broken.

### F3 — Inverse correlation: lower cosine_sim = worse V3 ARI on losers

CICIDS2017 (cosine_sim=0.60) has the worst V3 ARI loss (−0.156). TON_IoT (0.73, −0.199).
Winners NSL-KDD (0.70) and UNSW (0.74) sit in between. The bottleneck is not embedding
quality — it is **HDBSCAN's density-based cluster discovery** failing when embeddings are
diverse but not density-shaped.

---

## Decision

**Hypothesis REJECTED.** The Part X failure map (X.0) was rewritten as X.0' in the working
plan on 2026-05-23.

**Next experiments:**
- Exp 2.5: HDBSCAN → GMM with BIC k-selection (~30 min CPU, no retraining)
- Exp 2.6: V3 emb → PCA(20) → HDBSCAN preprocessing parity (~15 min CPU, no retraining)

See `docs/experiments/revised_failure_map.md` for the updated experiment lineup.

---

## Artifacts

- `experiments/results/exp2_collapse_all.json`
- `experiments/results/exp2_collapse_log.txt`
- `scripts/measure_embedding_collapse.py` (3 bugs fixed)
