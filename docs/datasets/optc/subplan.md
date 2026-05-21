# DARPA OpTC — Implementation Subplan (Part IV.6)

> Master plan reference: `Master_Plan.md`, Part IV.6

## Stage 1 — Audit

- [x] Read A: `hgnn/hgnn_correlation.py` — confirm `use_geometric_confidence` behavior
- [x] Read B: `datasets/loaders/darpa_optc_loader.py` — confirm column renaming
- [x] Create `scripts/audit_optc_labels.py`
- [x] Run audit → `docs/datasets/optc/audit.md`
- [x] Create scaffolding: README.md, subplan.md, decision_log.md, learnings.md

**Checkpoint decision**: GAEC overrides → `hgnn_checkpoints/multidomain_v2_optc_finetuned/best_supervised.pt`

## Stage 2 — Protocol Freeze

- [ ] Append OpTC blocks to `benchmark/datasets_real.yaml`
- [ ] Smoke test: `python -m benchmark.run_benchmark --datasets benchmark/datasets_real.yaml --output benchmark/results/latest/optc/smoke.csv`
- [ ] Confirm split files: `optc_10000_seed42.{npy,json}` and `optc_10000_seed142.{npy,json}`
- [ ] Write `docs/datasets/optc/protocol.md`

## Stage 3 — Baseline Roster

- [ ] Run: `python -m benchmark.run_benchmark --datasets benchmark/datasets_real.yaml --output benchmark/results/latest/optc/baseline_roster.csv`
- [ ] Expect 60 rows (10 methods × 3 seeds × 2 tracks)
- [ ] Note: standard ARI ~0.05 for ALL methods (correct, not a failure)
- [ ] Note: binary_ARI = headline metric

## Stage 4 — Decision Gate

- [ ] Compare V3 binary_ARI on campaign_id track
- [ ] Path A (wins by >0.1): proceed to Stage 5
- [ ] Path B (loses/degrades): write `investigation.md` before retuning
- [ ] Record decision in `decision_log.md`

## Stage 5 — Metrics + Demotion

- [ ] dominant_confusion_accuracy — pre-demote (degenerate on all 6 datasets)
- [ ] Standard ARI — retain but annotate as structurally low on 2-class
- [ ] binary_ARI on tactic — document as NaN (expected)
- [ ] Confirm all 12 metrics present in results.csv

## Stage 6 — Freeze v1.0

- [ ] Copy 11 artifacts to `benchmark/results/frozen/optc/v1.0/`
- [ ] Write `docs/datasets/optc/v1.0_baseline.md`
- [ ] Write `tests/test_optc_v1_frozen.py`
- [ ] Gate: `pytest -q` → 65+ passed, 0 errors
- [ ] Commit with message and tag
