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
| (queued) | — | `docs/datasets/ton_iot/` |

---

## CICIDS2017

| Date | Entry | Pointer |
|------|-------|---------|
| (queued) | — | `docs/datasets/cicids2017/` |

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
