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
| (pending) | Stage 1 — Audit | `docs/datasets/unsw_nb15/audit.md` |

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
