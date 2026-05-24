# MITRE-CORE — Claude Code Session Context

> ## ⚠️ 2026-05-24 AUDIT FINDING — Read first
>
> The V3 baseline (`network_v9_v3` at `alert_feature_dim=6`) feeds **two label-derived
> features as inputs at training AND inference time**: `tactics` (from `df['tactic']`)
> and `alert_types` (binary collapse of `df['alert_type']`). Both columns are also used
> as evaluation labels — this is input/label leakage.
>
> **Materiality test (Stage B, 2026-05-24)** — shuffling the two leaked columns at
> inference time produces these honest ARIs:
>
> | Dataset | Reported | Clean | Δ |
> |---|---|---|---|
> | NSL-KDD | 0.602 | 0.093 | −0.509 |
> | UNSW-NB15 | 0.564 | 0.289 | −0.275 |
> | TON-IoT v1.0 | 0.423 | 0.350 | −0.073 |
> | **TON-IoT v1.1 (GMM)** | **0.604** | **0.593** | **−0.011 (inert — GENUINE)** |
> | CICIDS2017 | 0.177 | 0.186 | +0.009 (inert) |
> | SQTK_SIEM v1.0 | 0.355 | 0.188 | −0.167 |
> | SQTK_SIEM v1.1 (pca=11) | 0.461 | 0.162 | −0.299 (regresses vs v1.0 clean) |
> | DARPA OpTC | 0.999 binary | −0.020 | −1.019 |
>
> **What is honest:** TON-IoT v1.1 (GMM+BIC clustering) and CICIDS2017 v1.0.
> Everything else needs reframing or re-running with a clean baseline.
>
> See `experiments/results/leakage_materiality.csv` for raw evidence,
> `docs/plans/MASTER_PLAN_v1.2.md` for the corrected program plan, and
> `docs/audits/v1.0_input_feature_audit.md` (planned) for the full audit report.
>
> The frozen artifacts (`benchmark/results/frozen/<dataset>/v1.x/`) are immutable
> historical records of what the pipeline produced — they document the leakage,
> they don't paper over it.

---

## What this project is (honest version)

A heterogeneous GNN trained with a topological contrastive objective that learns
a 128-dim embedding space for SOC alert correlation. **As of 2026-05-24, the
v1.0/v1.1 baseline numbers are known to be inflated by input feature leakage**
(see audit banner above). The architecture and infrastructure are real; the
specific ARI numbers reported pre-2026-05-24 are upper bounds, not validated
performance claims.

Two findings DO survive the audit:
1. **TON-IoT v1.1** (GMM+BIC clustering, 0.604 reported / 0.593 clean) — V3 is
   genuinely competitive with K-Means(raw) 0.622 on IoT traffic.
2. **CICIDS2017 v1.0** (0.177 reported / 0.186 clean) — the failure to discover
   BENIGN sub-structure is real and reproduces under clean measurement.

Everything else needs revalidation against `v2.0` clean baselines (planned).

## Two distinct problems

### 1. The system works (real)
- Ingests raw SIEM/IDS alerts and groups them into campaign clusters
- Genuine zero-shot performance on network IDS data (NSL-KDD ~0.74, UNSW ~0.54)
- Working confidence mechanism (GAEC) — not just softmax max-prob
- Runs in under 2 seconds on 2,000-alert batches
- Architecturally clean — 5 checkpoints, no dead code, documented edge types

### 2. The validation story is missing (data problem, not system problem)
- No controlled ablation experiments were run
- Design decisions (single layer, 6-dim features, GAT over GCN) are backed by reasoning and observed behavior, not experiments
- The ablation CSVs were fabricated placeholders — now archived
- The paper figures were hardcoded — now archived
- Closing this gap requires running experiments, not rewriting code

## Current state (v2.50, 2026-05-24 — Audit Finding)
v1.0/v1.1 frozen baselines retained as historical record. Leakage discovered 2026-05-24
in baseline feature extraction. v2.0 clean-baseline retrain queued. See `docs/audits/`,
`docs/plans/MASTER_PLAN_v1.2.md`, and `experiments/results/leakage_materiality.csv`.

## Canonical checkpoints

| Checkpoint | Best For | Reported ARI | Clean ARI | Mode |
|-----------|----------|--------------|-----------|------|
| `network_v9_v3/network_it_best.pt` | All 5 multi-class IDS datasets | 0.602 (NSL-KDD) | 0.093 (NSL-KDD) | Zero-shot GAEC |
| `siem_supcon_v4/best.pt` | SQTK_SIEM | 0.355 (v1.0) | 0.188 | GAEC |
| `unsw_supcon_v7/best.pt` | UNSW-NB15 semi-supervised | 0.538 | not tested | SupCon + Spectral |
| `multidomain_v2/best_supervised.pt` | Historical reference | 0.665 (UNSW) | not tested | Supervised softmax |
| `multidomain_v2_optc_finetuned/best_supervised.pt` | OpTC binary | 0.897 | not tested (suspect heavy leakage given OpTC clean=−0.02) | Supervised softmax |

All checkpoints share the same `AlertToGraphConverter._encode_alert_features` at
inference time, which produces the leaked dims 0 and 1. The leakage is in the
INPUT pipeline, not the checkpoint itself. Replacing the converter to skip the
leaked columns produces the "Clean ARI" values.

## Verified results

**Reported ARIs (with leakage; pre-2026-05-24 measurements):**

| Dataset | network_v9_v3 baseline | TON-IoT v1.1 (GMM) | SQTK_SIEM v1.1 (pca=11) |
|---|---|---|---|
| NSL-KDD | 0.602 | — | — |
| UNSW-NB15 | 0.564 | — | — |
| TON-IoT | 0.423 (v1.0) | **0.604** (v1.1) | — |
| CICIDS2017 | 0.177 | — | — |
| SQTK_SIEM | 0.355 (v1.0) | — | 0.461 (v1.1 — see note) |
| DARPA OpTC | 0.999 binary | — | — |

**Honest ARIs (Stage B materiality test, leaked features shuffled at inference):**

| Dataset | Reported | Clean | Δ |
|---|---|---|---|
| NSL-KDD | 0.602 | 0.093 | −0.509 |
| UNSW-NB15 | 0.564 | 0.289 | −0.275 |
| TON-IoT v1.0 | 0.423 | 0.350 | −0.073 |
| **TON-IoT v1.1** | 0.604 | 0.593 | **−0.011 (inert)** |
| CICIDS2017 | 0.177 | 0.186 | +0.009 (inert) |
| SQTK_SIEM v1.0 | 0.355 | 0.188 | −0.167 |
| SQTK_SIEM v1.1 | 0.461 | 0.162 | −0.299 (regresses vs v1.0 clean) |
| DARPA OpTC | 0.999 | −0.020 | −1.019 |

Use the clean column when comparing to honest baselines. Use the reported column
only when comparing to other pre-audit measurements made with the same pipeline.

## Claimed vs actual (revised 2026-05-24)

| Dimension | Pre-audit claim | Post-audit reality |
|-----------|-----------------|--------------------|
| Zero-shot universality | Works on 5/5 (later 2/6) network IDS | Works (clean) on TON-IoT (v1.1) and CICIDS2017. NSL-KDD/UNSW/OpTC "wins" were largely leakage. |
| Training mechanism | Hybrid topological + SimCLR (real) | Unchanged |
| Ablation validation | 7 validated design decisions | 0 validated; baseline feature description was demonstrably false |
| NSL-KDD zero-shot | 0.739 → 0.602 (v1.0) | 0.093 clean (cannot separate attack types without the tactic input) |
| "Purely unsupervised" | True for 2/6 datasets | False for ALL 6 — the input pipeline saw labels |
| TON-IoT v1.1 GMM improvement | +0.181 ARI | +0.170 honest (0.423 → 0.593) — biggest validated result in the program |
| SQTK_SIEM v1.1 pca=11 improvement | +0.106 ARI | −0.026 honest (0.188 → 0.162) — invalidated |

## What closing the gap looks like

| Task | Effort | Impact |
|------|--------|--------|
| Run real ablation sweeps (UF on/off, layer count) | ~1 week | Validates/invalidates documented design decisions |
| Retrain a 2-layer variant | ~2 hours GPU | Tests single-layer claim empirically |
| Run NSL-KDD/UNSW at 10K samples consistently | ~1 day | Locks down headline numbers |
| Honest zero-shot framing ("works on network IDS") | Documentation edit | No experiments needed |
| Replace fabricated figures with sweep-generated plots | ~1 day coding | Makes the paper story real |

## Key files

- `training/train_graph_mae_v9_multidata_fast.py` — SSL training (topological + SimCLR hybrid)
- `training/finetune_supcon.py` — SupCon fine-tuning (has its own inline SupConLoss)
- `hgnn/hgnn_correlation.py` — main engine (HGNNCorrelationEngine + MITREHeteroGNN)
- `experiments/run_gate_tuning.py` — sweep runner
- `MEMORY.md` — full versioned experiment history (v2.2 → v2.40)

## Confirmed findings across all versions

- `use_uf_refinement=False` is correct default (UF is net-harmful) — STILL VALID
- GAEC mode > softmax mode for clustering — STILL VALID
- ~~Single GAT layer avoids over-smoothing~~ Single-layer choice is independent of
  the leakage finding; Exp 2 S1 confirmed cosine_sim ≈ 0.7 across all datasets, no
  collapse exists.
- ~~6-dim base features are domain-agnostic and generalize better than 15-dim contextual~~
  **FALSE.** The actual baseline 6-dim is `tactics, alert_types, hour, dow, protocols,
  services` (per `hgnn/hgnn_correlation.py:902-907`) — includes two label-derived
  features. The "domain-agnostic" framing was inaccurate. Corrected 2026-05-24.
- Bridge edges and entity collapse: both zero effect — STILL VALID
- CS fine-tuning ineffective — STILL VALID
- HDBSCAN seeding fixed — full reproducibility with seed=42 — STILL VALID
- **(NEW 2026-05-24)** GMM+BIC clustering reproducibly improves TON-IoT under both
  leaked and clean inputs (Δ from clean baseline: +0.243, i.e. 0.350 → 0.593). Pure
  inference change, no retraining.
- **(NEW 2026-05-24)** PCA component count sweep on SQTK_SIEM produces leakage-driven
  artifacts; pca=11 v1.1 freeze does not reproduce honestly. Lesson: future sweeps
  must be validated against the clean baseline before freezing.
