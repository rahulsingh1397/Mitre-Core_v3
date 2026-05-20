---
Master plan: MASTER_PLAN_v1.0.md
Section implemented: Part IV.3 — TON_IoT
Lifecycle stages covered: 1–6
---

# TON_IoT — Subplan v1.0

## Stage 1 — Audit ✅
- [x] Run `scripts/audit_ton_iot_labels.py`
- [x] Record in `audit.md`: row count, SHA256, schema diff, label tracks, graph feasibility
- [x] Identify at least one usable multi-class label track
- **Exit criterion:** label audit written; at least one usable track confirmed ✅

**Findings:**
- 211,043 rows; parquet format; SHA256 `0d307cb86b64099efb13088d94096a7863f3b5500396887eab437fb88ca0ce6f`
- Missing `hostname`/`username` vs NSL-KDD/UNSW — graph runs fine (smoke tested: 29 clusters, 0% noise)
- Selected tracks: `alert_type` (primary, 10 classes), `tactic` (secondary, 7 after NaN-fill), `label` (binary int, benign_label=0)
- `campaign_id` excluded (1-to-1 duplicate of alert_type)
- mitm class: 1,043 rows (~49 in 10K subset) — sparse but evaluable

## Stage 2 — Protocol Freeze
- [ ] Sample size: 10,000 (default, matching NSL-KDD/UNSW)
- [ ] Add YAML block to `benchmark/datasets_real.yaml` (TON-IoT-dev seed 42, TON-IoT eval seed 142)
      Note: use `benign_label: 0` (int) since `label` column is binary int, not `alert_type`="normal"
      Actually: primary label col is `alert_type` (text); `benign_label` applies to the binary `label` track
- [ ] Run smoke benchmark on dev subset → splits persisted to `benchmark/splits/`
- [ ] Verify splits are disjoint
- **Exit criterion:** splits on disk, disjoint, hashes recorded, manifest emitted

## Stage 3 — Baseline Roster
- [ ] Run all 10 baselines + V3 on eval subset, seeds 42/43/44
- [ ] Output to `benchmark/results/latest/ton_iot/baseline_roster.csv`
- **Exit criterion:** all 10 rows produced, no method crashed

## Stage 4 — Decision Gate
- [ ] Compare V3 ARI vs best baseline on each label track
- [ ] **Path A (margin >0.1 ARI):** skip sweep → Stage 5
- [ ] **Path B (margin ≤0.1 or V3 loses):** write `investigation.md` first,
      then sweep via `benchmark/clustering_sweep_full_engine.py` (NOT standalone)
- [ ] Record choice in `decision_log.md`
- **Honest expectation:** V3 may not dominate on IoT (per master plan). Publish either way.
- **Exit criterion:** decision recorded; either sweep complete or Path A confirmed

## Stage 5 — Metrics + Demotion
- [ ] Confirm 12-metric set emitted (inc. attack_f1_demoted, n_pred_clusters, noise_fraction)
- [ ] Flag any degenerate metric (constant across methods)
- [ ] Note mitm class behavior (~49 rows in 10K subset)
- **Exit criterion:** full metric set in results CSV, degenerate metrics documented

## Stage 6 — Freeze v1.0
- [ ] Copy artifacts → `benchmark/results/frozen/ton_iot/v1.0/`
- [ ] Write `v1.0_baseline.md`
- [ ] Add `tests/test_ton_iot_v1_frozen.py`
- [ ] Commit + git tag `ton-iot-v1.0`
- [ ] Append one-line entry to `IMPROVEMENT_LOG.md`
- [ ] Fill `learnings.md` before moving to next dataset
- **Exit criterion:** all 5 subplan exit criteria in master plan section VII.5 satisfied
