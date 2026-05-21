---
Master plan: MASTER_PLAN_v1.0.md
Section implemented: Part IV.4 — CICIDS2017
Lifecycle stages covered: 1–6
---

# CICIDS2017 — Subplan v1.0

## Stage 1 — Audit
- [x] Run `scripts/audit_cicids2017_labels.py`
- [x] Record in `audit.md`: row count, SHA256, schema diff, label tracks, graph feasibility
- [x] Identify at least one usable multi-class label track
- **Exit criterion:** label audit written; at least one usable track confirmed ✅

## Stage 2 — Protocol Freeze
- [x] Sample size: 10,000 (default, matching NSL-KDD/UNSW/TON_IoT)
- [x] Add YAML blocks to `benchmark/datasets_real.yaml` (CICIDS2017-dev seed 42, CICIDS2017 seed 142)
- [x] Run smoke benchmark on dev subset → splits persisted to `benchmark/splits/`
- [x] Verify splits are disjoint (overlap=0 confirmed)
- **Exit criterion:** splits on disk, disjoint, hashes recorded, manifest emitted ✅

## Stage 3 — Baseline Roster
- [x] Run all 10 baselines + V3 on eval subset, seeds 42/43/44
- [x] Output to `benchmark/results/latest/cicids2017/baseline_roster.csv`
- **Exit criterion:** all 10 rows produced, no method crashed ✅

## Stage 4 — Decision Gate
- [x] Compare V3 ARI vs best baseline on alert_type track
- [x] Path B: V3 ARI=0.111 vs Spectral(emb) ARI=0.333 — margin=-0.222
- [x] Write investigation.md
- [x] Run full-engine sweep (scripts/sweep_cicids2017.py, 168 configs)
- [x] Evaluate sweep winner (mcs=200, pca=8, eps=0.15) on eval subset (3 seeds)
- [x] Decision: freeze with sweep winner; V3 ARI=0.177 (2nd); V3 loses by -0.156 ARI
- **Exit criterion:** decision recorded in decision_log.md ✅

## Stage 5 — Metrics + Demotion
- [x] All 12 metrics present in results.csv
- [x] `dominant_confusion_accuracy` pre-demoted (confirmed degenerate on 4/4 datasets now)
- [x] DBSCAN demoted: attack_f1_demoted=0.000 (n_pred_clusters=2, degenerate binary split)
- **Exit criterion:** full metric set, degenerate metrics documented ✅

## Stage 6 — Freeze v1.0
- [x] Copy artifacts → `benchmark/results/frozen/cicids2017/v1.0/` (11 files: results, summary, manifest, engine_config, checkpoint_sha256, dataset_sha256, environment, splits ×2)
- [x] Write `v1.0_baseline.md`
- [x] Add `tests/test_cicids2017_v1_frozen.py` (frozen ARI=0.1771, AMI=0.5699)
- [ ] `pytest -q` gate (pending)
- [ ] Commit + git tag `cicids2017-v1.0` (pending)
- [ ] Append entries to `IMPROVEMENT_LOG.md` (pending)
- [x] Fill `learnings.md`
- **Exit criterion:** all subplan exit criteria satisfied (pending pytest gate + commit)
