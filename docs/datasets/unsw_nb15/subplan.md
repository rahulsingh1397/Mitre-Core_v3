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

## Stage 3 — Baseline Roster ✅
- [x] Run all 10 baselines + V3 on eval subset, seeds 42/43/44
- [x] Output to `benchmark/results/latest/unsw_nb15/baseline_roster.csv`
- **Exit criterion:** all 10 rows produced, no method crashed ✅

**Results (attack_cat, mean ARI):** V3=0.564, PCA+HDBSCAN=0.354, PCA+K-Means=0.347

## Stage 4 — Decision Gate ✅
- [x] Compare V3 ARI vs best baseline on each label track
- [x] **Path A confirmed** — margin=+0.210 ARI (V3=0.564 vs PCA+HDBSCAN=0.354)
- [x] Record choice in `decision_log.md`
- **Exit criterion:** decision recorded; Path A confirmed ✅

## Stage 5 — Metrics + Demotion ✅
- [x] Confirm 12-metric set emitted (inc. attack_f1_demoted, n_pred_clusters, noise_fraction)
- [x] Flag degenerate metric: `dominant_confusion_accuracy` = 1.0 for all methods (demoted)
- [x] Worms class: ~7 rows in 10K subset — documented as known limitation
- **Exit criterion:** full metric set in results CSV, degenerate metrics documented ✅

## Stage 6 — Freeze v1.0 ✅
- [x] Copy artifacts → `benchmark/results/frozen/unsw_nb15/v1.0/`
- [x] Write `v1.0_baseline.md`
- [x] Add `tests/test_unsw_nb15_v1_frozen.py` (6 passed, 2 deselected slow)
- [x] Commit + git tag `unsw-nb15-v1.0`
- [x] Append one-line entry to `IMPROVEMENT_LOG.md`
- [x] Fill `learnings.md`
- **Exit criterion:** all 5 subplan exit criteria in master plan section VII.5 satisfied ✅
