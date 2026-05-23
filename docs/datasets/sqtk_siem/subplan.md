# SQTK_SIEM — Subplan (Part IV.5)

**Master plan:** `docs/plans/MASTER_PLAN_v1.0.md` Part IV.5
**Dataset:** `datasets/SQTK_SIEM/mitre_core_format.csv`
**Checkpoint:** `hgnn_checkpoints/siem_supcon_v4/best.pt`
**Planned tag:** `sqtk-siem-v1.0`

---

## Protocol Deviations from Prior Datasets

1. **No disjoint dev/eval split** — 5,100 rows < sample_size=10,000. Using `sample_size: 5100` with no `exclude_sample_seeds`. All 3 benchmark seeds (42/43/44) run on the full corpus.
2. **Different checkpoint** — `siem_supcon_v4/best.pt` instead of `network_v9_v3`. First dataset-specific checkpoint in the benchmark.
3. **9 frozen artifacts** — No seed142 split pair (no disjoint split).
4. **n_clusters = 14** — Matches full alert_type class count.
5. **kcluster NOT used as track** — Pre-computed from prior analysis; documented as reference only.

---

## Stage 1 — Audit ✅

Script: `scripts/audit_sqtk_siem_labels.py` (inline via temp script)

- 5,100 rows, 21 columns
- alert_type: 14 classes, 0 nulls — primary track
- tactic: 9 classes, UNKNOWN dominates (88.7%) — secondary track
- campaign_id: 8 classes, UNKNOWN dominates (89.4%) — secondary track
- kcluster: 11 clusters — NOT an evaluation track

Exit criterion: ✅ At least one usable multi-class track confirmed (alert_type, 14 classes).

---

## Stage 2 — Protocol Freeze ✅

YAML block added to `benchmark/datasets_real.yaml`.

Exit criterion: ✅ Splits persisted, hashes recorded.

---

## Stage 3 — Baseline Roster ✅

Run all 10 baselines + V3 on the full 5,100 rows, seeds 42/43/44.

Command: `python -m benchmark.run_benchmark --datasets benchmark/datasets_real.yaml --output benchmark/results/latest/sqtk_siem/baseline_roster.csv`

Exit criterion: ✅ All rows produced (90 rows = 10 methods × 3 seeds × 3 tracks); checkpoint loading bug fixed (`alert_feature_dim` mismatch).

---

## Stage 4 — Decision Gate ✅ PATH B

- V3 ARI = 0.3551 ± 0.0000
- Best baseline ARI = PCA + HDBSCAN 0.3825 ± 0.0000
- Margin = −0.027 (V3 loses)
- **Path B taken:** `investigation.md` written before any retuning. ~~Root cause: embedding collapse/over-smoothing (mean cosine similarity = 0.958).~~ **[CORRECTION 2026-05-23: the cosine_sim figure was a measurement artifact — `alert_feature_dim` hardcoded to 6 in the diagnostic script; actual value = 0.79. Root cause reclassified to clustering-algorithm / preprocessing gap. investigation.md has been invalidated with a banner. See `docs/experiments/multi_layer_depth.md`.]**

---

## Stage 5 — Metrics + Demotion ✅

- `dominant_confusion_accuracy` — pre-demoted (constant 1.0, fifth consecutive dataset).
- `attack_f1` — degenerate (no BENIGN class in SQTK_SIEM).
- All 12 metrics present in results.csv.

---

## Stage 6 — Freeze v1.0 ✅

1. ✅ Create `benchmark/results/frozen/sqtk_siem/v1.0/` with 9 files.
2. ✅ Write `docs/datasets/sqtk_siem/v1.0_baseline.md`.
3. ✅ Write `docs/datasets/sqtk_siem/protocol.md`.
4. ✅ Add `tests/test_sqtk_siem_v1_frozen.py`.
5. ✅ `pytest -q` gate → 23 passed, 2 deselected, 0 errors.
6. ⏳ Commit + tag `sqtk-siem-v1.0` (user to run).
7. ✅ Update `IMPROVEMENT_LOG.md`.
8. ✅ Fill `docs/datasets/sqtk_siem/learnings.md`.

---

## Files Created/Touched

**Created:**
- `docs/datasets/sqtk_siem/{README,subplan,audit,protocol,decision_log,learnings,investigation,v1.0_baseline}.md`
- `benchmark/splits/sqtk_siem_5100_seed42.{npy,json}`
- `benchmark/results/latest/sqtk_siem/*`
- `benchmark/results/frozen/sqtk_siem/v1.0/*` (9 files)
- `tests/test_sqtk_siem_v1_frozen.py`

**Touched:**
- `benchmark/datasets_real.yaml` — append SQTK_SIEM block
- `hgnn/hgnn_correlation.py` — fix `alert_feature_dim` loading from checkpoint
- `tests/test_split_disjoint.py` — add single-split test
- `IMPROVEMENT_LOG.md`
