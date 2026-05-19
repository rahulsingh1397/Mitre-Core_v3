# HGNN Checkpoint Cleanup Report

**Status: ✅ EXECUTED (May 2026)**

## Objective
Analyze `hgnn_checkpoints/` and reconcile listed-vs-actual artifacts: keep
checkpoints that are referenced by active scripts, remove orphaned configs,
and rewrite the README to match reality.

## 1. Surviving Active Checkpoints (Post-Cleanup)
The directories below contain valid `.pt` files and are referenced by at least
one active script in `experiments/`, `training/`, or `hgnn/`.

| Directory | File | Referenced by |
|-----------|------|---------------|
| `network_v9_v3/` | `network_it_best.pt` | `experiments/run_gate_tuning.py` (canonical), `experiments/run_baseline_clustering.py`, `experiments/run_multiseed_headline_table.py` |
| `unsw_supcon_v7/` | `best.pt` | UNSW-NB15 `checkpoint_override` in `DATASET_CONFIG` |
| `siem_supcon_v4/` | `best.pt` + `test_indices.npy` | SQTK_SIEM_kcluster `checkpoint_override` in `DATASET_CONFIG` |
| `multidomain_v2/` | `best_supervised.pt` | `experiments/run_cross_dataset_generalization.py` |
| `multidomain_v2_optc_finetuned/` | `best_supervised.pt` + `training_history.json` | OpTC fine-tuning experiments |
| `archive/` | — | Historical reference only (not loaded by active code) |

## 2. Retired Directories (Verified Absent)
The following directories listed in the original report no longer exist on disk;
their model files were either moved to `archive/` or deleted in earlier cleanup
passes:

* `unsw_nb15_retrained/` — the "ARI=0.8994 breakthrough" claim was single-seed
  and depended on a supervised head incompatible with pure-unsupervised inference.
* `multidomain_v2_mitre_core_format_finetuned/`
* `multidomain_v2_nslkdd_finetuned/`
* `multidomain_v2_siem_finetuned/`
* `multidomain_unsw_beth/`
* `beth_retrained/`
* `optc_retrained/`

## 3. Orphaned JSON Configs (All Removed)
| File | Status |
|------|--------|
| `nsl_kdd_best_config.json` | ✅ Already deleted prior to this report |
| `nsl_kdd_hgnn_stats.json` | ✅ **Deleted in this cleanup pass (May 2026)** |
| `nsl_kdd_optuna_best_config.json` | ✅ Already deleted |
| `unsw_finetuned_config.json` | ✅ Already deleted |
| `unsw_nb15_best_config.json` | ✅ Already deleted |
| `unsw_nb15_hgnn_stats.json` | ✅ Already deleted |
| `unsw_supervised_config.json` | ✅ Already deleted |

## Action Plan — Completed
1. ✅ Deleted remaining orphaned JSON (`nsl_kdd_hgnn_stats.json`).
2. ✅ Rewrote `hgnn_checkpoints/README.md` to list only surviving directories,
   set the canonical default to `network_v9_v3/network_it_best.pt`, and document
   the per-dataset checkpoint overrides used by `run_gate_tuning.py`.
3. ✅ This report updated to reflect executed state.
