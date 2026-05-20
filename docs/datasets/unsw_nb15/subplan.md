---
Master plan: MASTER_PLAN_v1.0.md
Section implemented: Part IV.2 — UNSW-NB15
Lifecycle stages covered: 1–6
---

# UNSW-NB15 — Subplan v1.0

## Stage 1 — Audit ✅
- [x] Run `scripts/audit_unsw_nb15_labels.py`
- [x] Record in `audit.md`: row count, SHA256, column list, value counts per label track, flagged anomalies
- [x] Identify at least one usable multi-class label track
- **Exit criterion:** label audit written; at least one usable track confirmed ✅

**Findings:**
- 175,341 rows; SHA256 `c7856d8428fd7b35ffd233ccece378be3e0b2ba9d23c6b7bfe37dab13441b892`
- `tactic` has 56,000 NaN for normal rows (same pattern as NSL-KDD; benign_label handles it)
- Selected tracks: `attack_cat` (primary, 10 classes), `tactic` (secondary, 8 after NaN-fill), `alert_type` (binary)
- Excluded: `campaign_id` (redundant with attack_cat), `stage` (temporal phases)
- Warning: Worms class has only 130 rows in full corpus (~7 in 10K subset)

## Stage 2 — Protocol Freeze
- [ ] Decide sample size (default 10,000 — confirmed; document Worms caveat)
- [ ] Add YAML block to `benchmark/datasets_real.yaml` (UNSW-NB15-dev seed 42, UNSW-NB15 eval seed 142)
- [ ] Run smoke benchmark on dev subset → splits persisted to `benchmark/splits/`
- [ ] Verify splits are disjoint
- **Exit criterion:** splits on disk, disjoint, hashes recorded, manifest emitted

## Stage 3 — Baseline Roster
- [ ] Run all 10 baselines + V3 on eval subset, seeds 42/43/44
- [ ] Output to `benchmark/results/latest/unsw_nb15/baseline_roster.csv`
- **Exit criterion:** all 10 rows produced, no method crashed

## Stage 4 — Decision Gate
- [ ] Compare V3 ARI vs best baseline on each label track
- [ ] **Path A (margin >0.1 ARI):** skip sweep → Stage 5
- [ ] **Path B (margin ≤0.1 or V3 loses):** write `investigation.md` first,
      then sweep via `benchmark/clustering_sweep_full_engine.py` (NOT standalone)
- [ ] Record choice in `decision_log.md`
- **Exit criterion:** decision recorded; either sweep complete or Path A confirmed

## Stage 5 — Metrics + Demotion
- [ ] Confirm 12-metric set emitted (inc. attack_f1_demoted, n_pred_clusters, noise_fraction)
- [ ] Flag any degenerate metric (constant across methods)
- [ ] Note Worms class behavior (expected poor cluster recall given ~7 rows in 10K subset)
- **Exit criterion:** full metric set in results CSV, degenerate metrics documented

## Stage 6 — Freeze v1.0
- [ ] Copy artifacts → `benchmark/results/frozen/unsw_nb15/v1.0/`
- [ ] Write `v1.0_baseline.md` (from template at NSL-KDD's baseline doc)
- [ ] Add `tests/test_unsw_nb15_v1_frozen.py`
- [ ] Commit + git tag `unsw-nb15-v1.0`
- [ ] Append one-line entry to `IMPROVEMENT_LOG.md`
- [ ] Fill `learnings.md` before moving to next dataset
- **Exit criterion:** all 5 subplan exit criteria in master plan section VII.5 satisfied
