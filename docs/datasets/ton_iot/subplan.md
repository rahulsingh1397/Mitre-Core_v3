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

## Stage 3 — Baseline Roster ✅
- [x] Run all 10 baselines + V3 on eval subset, seeds 42/43/44
- [x] Output to `benchmark/results/latest/ton_iot/baseline_roster.csv`
- **Exit criterion:** all 10 rows produced, no method crashed ✅

**Results (alert_type, mean ARI):** K-Means(raw)=0.622, V3=0.423

## Stage 4 — Decision Gate ✅ (Path B)
- [x] V3 ARI=0.423 vs K-Means(raw)=0.622 — margin=-0.199 → **PATH B**
- [x] Wrote `investigation.md` before sweep
- [x] Ran targeted mcs sweep (full engine, dev split)
- [x] Swept winner mcs=300 evaluated on eval: ARI=0.474 (gap=0.148 > honest cap 0.05)
- [x] Freeze with Path B result — V3 loses on TON_IoT (documented finding)
- **Exit criterion:** decision recorded; sweep run; honest result frozen ✅

## Stage 5 — Metrics + Demotion ✅
- [x] All 12 metrics present
- [x] `dominant_confusion_accuracy` demoted (constant 1.0 — third dataset to confirm structural degeneracy)
- [x] mitm class: ~49 rows in 10K subset — sparse but represented
- **Exit criterion:** full metric set, degenerate metrics documented ✅

## Stage 6 — Freeze v1.0 ✅
- [x] Copy artifacts → `benchmark/results/frozen/ton_iot/v1.0/` (11 files)
- [x] Write `v1.0_baseline.md` (with honest "V3 loses" headline)
- [x] Add `tests/test_ton_iot_v1_frozen.py` (6 passed, 2 deselected slow)
- [x] Commit + git tag `ton-iot-v1.0`
- [x] Append one-line entry to `IMPROVEMENT_LOG.md`
- [x] Fill `learnings.md`
- **Exit criterion:** all 5 subplan exit criteria satisfied ✅
