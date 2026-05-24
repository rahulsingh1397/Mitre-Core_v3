# MITRE-CORE V3 — Improvement Log (Index)

Each entry is one line. Full details live in `docs/datasets/<name>/` per the per-dataset lifecycle.
For the pre-freeze NSL-KDD execution history (Phases 0–5), see `docs/archive/v2/` or the git log.

---

## NSL-KDD

| Date | Entry | Pointer |
|------|-------|---------|
| 2026-05-17 | Baseline assessment: real-dataset benchmark wiring established | `docs/datasets/nsl_kdd/decision_log.md` |
| 2026-05-17 | Phase 1 — Protocol hardening: frozen eval split (seed 142), disjoint dev/eval, manifest emission | `docs/datasets/nsl_kdd/protocol.md` |
| 2026-05-18 | Phase 2 — Label-track validation: tactic + alert_type + campaign_id tracks confirmed | `docs/datasets/nsl_kdd/audit.md` |
| 2026-05-19 | Phase 3 — Sweep methodology lesson: standalone HDBSCAN winner does not transfer to full engine | `docs/lessons/phase3_sweep_methodology.md` |
| 2026-05-19 | Phase 4 — Stronger baselines: embedding-based baselines added (K-Means/Spectral/HDBSCAN on HGNN emb) | `docs/datasets/nsl_kdd/decision_log.md` |
| 2026-05-19 | Phase 5 — Full 12-metric set + attack_f1_demoted; Spectral(raw) revealed as strongest raw baseline | `docs/datasets/nsl_kdd/decision_log.md` |
| 2026-05-19 | **v1.0 FROZEN** — V3 ARI=0.602 ± 0.000 (tactic), margin +0.188 ARI over Spectral(raw) | `docs/datasets/nsl_kdd/v1.0_baseline.md` |

---

## UNSW-NB15

| Date | Entry | Pointer |
|------|-------|---------|
| 2026-05-19 | Stage 1 — Audit: 175,341 rows, attack_cat primary (10 classes), tactic secondary, alert_type binary | `docs/datasets/unsw_nb15/audit.md` |
| 2026-05-19 | Stage 2 — Protocol freeze: 10K sample, seed 42 dev / seed 142 eval, disjoint | `docs/datasets/unsw_nb15/protocol.md` |
| 2026-05-19 | Stage 3 — Baseline roster: 10 methods × 3 seeds × 3 tracks complete | `benchmark/results/latest/unsw_nb15/baseline_roster.csv` |
| 2026-05-19 | Stage 4 — Path A: V3 ARI=0.564 vs best baseline 0.354 (+0.210 margin); no sweep | `docs/datasets/unsw_nb15/decision_log.md` |
| 2026-05-19 | Stage 5 — dominant_confusion_accuracy demoted (constant 1.0 across all methods) | `docs/datasets/unsw_nb15/decision_log.md` |
| 2026-05-19 | **v1.0 FROZEN** — V3 ARI=0.564 ± 0.000 (attack_cat), margin +0.210 ARI over PCA+HDBSCAN | `docs/datasets/unsw_nb15/v1.0_baseline.md` |

---

## TON_IoT

| Date | Entry | Pointer |
|------|-------|---------|
| 2026-05-20 | Stage 1 — Audit: 211,043 rows parquet, alert_type primary (10 classes), tactic secondary; schema diff vs NSL-KDD documented; graph feasibility confirmed (IP-only, 0% noise) | `docs/datasets/ton_iot/audit.md` |
| 2026-05-20 | Stage 2 — Protocol freeze: 10K sample, seed 42 dev / seed 142 eval, disjoint; parquet handled natively by benchmark.py | `docs/datasets/ton_iot/protocol.md` |
| 2026-05-20 | Stage 3 — Baseline roster: 10 methods × 3 seeds; K-Means(raw) leads at ARI=0.622 (n_clusters=10 privileged prior) | `benchmark/results/latest/ton_iot/baseline_roster.csv` |
| 2026-05-20 | Stage 4 — Path B: V3 ARI=0.423 vs K-Means 0.622 (−0.199); HDBSCAN over-segmentation (55 clusters for 10 classes) diagnosed; full-engine sweep (mcs=300 winner, eval ARI=0.474, gap=0.148 > honest cap) | `docs/datasets/ton_iot/investigation.md` |
| 2026-05-20 | Stage 5 — dominant_confusion_accuracy demoted (constant 1.0, third consecutive dataset — structurally degenerate) | `docs/datasets/ton_iot/decision_log.md` |
| 2026-05-20 | **v1.0 FROZEN** — V3 loses (ARI=0.423 ± 0.000); K-Means(raw) wins (ARI=0.622 ± 0.033); honest Path B result; git tag `ton-iot-v1.0` | `docs/datasets/ton_iot/v1.0_baseline.md` |

---

## CICIDS2017

| Date | Entry | Pointer |
|------|-------|---------|
| 2026-05-20 | Stage 1 — Audit: 3,119,345 rows parquet, 10 cols, alert_type primary (15 classes, 288,602 null rows), IP-only graph; 72.9% BENIGN imbalance documented | `docs/datasets/cicids2017/audit.md` |
| 2026-05-20 | Stage 2 — Protocol freeze: 10K sample, seed 42 dev / seed 142 eval, disjoint; null rows treated as "UNKNOWN" class via benchmark fillna; n_clusters=15 (full schema count) | `docs/datasets/cicids2017/protocol.md` |
| 2026-05-20 | Stage 3 — Baseline roster: 10 methods × 3 seeds; default config V3 ARI=0.111 (45 clusters); Spectral(emb) leads at ARI=0.333 | `benchmark/results/latest/cicids2017/baseline_roster.csv` |
| 2026-05-20 | Stage 4 — Path B: V3 ARI=0.111 vs Spectral(emb) 0.333 (−0.222); root cause: BENIGN over-segmentation (mcs=5 → 30 BENIGN sub-clusters); full-engine sweep (168 configs): winner mcs=200/pca=8/eps=0.15, ARI=0.177 (11 clusters); gap −0.156 remains | `docs/datasets/cicids2017/investigation.md` |
| 2026-05-20 | Stage 5 — dominant_confusion_accuracy demoted (constant 1.0, fourth consecutive dataset); DBSCAN demoted (n_clusters=2, attack_f1_demoted=0.000) | `docs/datasets/cicids2017/decision_log.md` |
| 2026-05-20 | **v1.0 FROZEN** — V3 ARI=0.177 ± 0.000, AMI=0.570 ± 0.000 (sweep winner, 11 clusters); V3 2nd on ARI, 1st on AMI; Spectral(emb) wins ARI; git tag `cicids2017-v1.0` | `docs/datasets/cicids2017/v1.0_baseline.md` |

---

## SQTK_SIEM

| Date | Entry | Pointer |
|------|-------|---------|
| 2026-05-21 | Stage 1 — Audit: 5,100 rows, 21 cols, alert_type primary (14 classes), tactic/campaign_id secondary (UNKNOWN 88–89%); kcluster (11) excluded as circular | `docs/datasets/sqtk_siem/audit.md` |
| 2026-05-21 | Stage 2 — Protocol freeze: full 5,100 rows, no disjoint split (corpus < sample_size); n_clusters=14; first dataset-specific checkpoint (`siem_supcon_v4/best.pt`) | `docs/datasets/sqtk_siem/protocol.md` |
| 2026-05-21 | Stage 3 — Baseline roster: 10 methods × 3 seeds × 3 tracks complete; checkpoint loading bug fixed (`alert_feature_dim` now read from checkpoint shape) | `benchmark/results/latest/sqtk_siem/baseline_roster.csv` |
| 2026-05-21 | Stage 4 — Path B: V3 ARI=0.355 vs PCA+HDBSCAN 0.382 (−0.027); root cause: embedding collapse/over-smoothing (mean cosine similarity=0.958); investigation written before retuning | `docs/datasets/sqtk_siem/investigation.md` |
| 2026-05-21 | Stage 5 — dominant_confusion_accuracy demoted (constant 1.0, fifth consecutive dataset); attack_f1 degenerate (no BENIGN class) | `docs/datasets/sqtk_siem/decision_log.md` |
| 2026-05-21 | **v1.0 FROZEN** — V3 ARI=0.355 ± 0.000, AMI=0.604 ± 0.000; V3 loses to PCA+HDBSCAN (ARI=0.382); honest Path B result; 9 frozen artifacts; git tag `sqtk-siem-v1.0` | `docs/datasets/sqtk_siem/v1.0_baseline.md` |

---

## DARPA OpTC

| Date | Entry | Pointer |
|------|-------|---------|
| 2026-05-21 | Stage 1 — Audit: 4,656,650 rows, 29 cols, CampaignId primary (binary: Benign 95.8% / RedTeam_Sep23 4.2%), tactic degenerate (Execution only); richest graph in benchmark (src_ip, dst_ip, hostname, username all present) | `docs/datasets/optc/audit.md` |
| 2026-05-21 | Stage 2 — Protocol freeze: 10K sample, seed 42 dev / seed 142 eval, disjoint; checkpoint = `multidomain_v2_optc_finetuned/best_supervised.pt` (GAEC overrides) | `docs/datasets/optc/protocol.md` |
| 2026-05-21 | Stage 3 — Baseline roster: 10 methods x 3 seeds x 2 tracks complete; standard ARI ~0.05 for all methods by design on 2-class dataset | `benchmark/results/latest/optc/baseline_roster.csv` |
| 2026-05-21 | Stage 4 — Path A: V3 binary_ARI=0.999 (tied with HDBSCAN emb), >0.1 margin over all baselines; no sweep | `docs/datasets/optc/decision_log.md` |
| 2026-05-21 | Stage 5 — dominant_confusion_accuracy demoted (sixth consecutive dataset, constant 1.0); standard ARI annotated as structurally low on 2-class; tactic track degenerate | `docs/datasets/optc/decision_log.md` |
| 2026-05-21 | **v1.0 FROZEN** — V3 binary_ARI=0.999, AMI=0.203, ARI=0.059 (campaign_id track); near-perfect binary separation; 11 frozen artifacts; git tag `darpa-optc-v1.0` | `docs/datasets/optc/v1.0_baseline.md` |

---

## Experiments

### V3 Ablation Record

Started 2026-05-23. Single source of truth for all Part X experiment results, replacing
the archived V2 placeholders. See `docs/ablations/v3_ablation_record.{csv,md}`.

Currently 12 rows backfilled from Exp 1 / 2.5 / 2.5b / 2.6 / 2.6b. Each subsequent
experiment appends one row per dataset benchmarked.

### v1.0 Integrity Audit — 2026-05-24

Triggered during Exp 3 (15-dim features) pre-flight. Stage A direct verification + Stage B materiality test confirmed that the v1.0 baseline feature extractor feeds two label-derived columns (`tactics`, `alert_types`) as model inputs. Shuffle test on all v1.0 + v1.1 freezes returned MATERIAL verdict.

**Two findings survive the audit:**
- TON-IoT v1.1 (GMM+BIC clustering): Δ_clean = −0.011 inert → GENUINE
- CICIDS2017 v1.0: Δ_clean = +0.009 inert → GENUINE (honest loser)

**All other v1.x freezes are upper bounds.** SQTK_SIEM v1.1 (pca=11) specifically REGRESSES under clean measurement (clean ARI 0.162 < v1.0 clean 0.188) — the pca=11 lift was a leakage interaction artifact.

Full report: `docs/audits/v1.0_input_feature_audit.md`.
Validity ledger: `docs/plans/MASTER_PLAN_v1.2.md`.
Raw evidence: `experiments/results/leakage_materiality.csv`.

Frozen artifacts preserved as immutable reproducibility records. Per-dataset baseline docs got correction banners. v2.0 clean-baseline retrain queued.
