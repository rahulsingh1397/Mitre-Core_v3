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
| (queued) | — | `docs/datasets/sqtk_siem/` |

---

## DARPA OpTC

| Date | Entry | Pointer |
|------|-------|---------|
| (queued) | — | `docs/datasets/darpa_optc/` |
