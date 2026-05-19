# HGNN Checkpoint Index

This directory contains the surviving checkpoints used by MITRE-CORE v2.43.
Legacy `v2.x`, `unsw_nb15_retrained`, `beth_retrained`, `multidomain_unsw_beth`,
and `multidomain_v2_*_finetuned` (except OpTC) directories were retired during
the May 2026 cleanup; their orphaned JSON configs were removed and their model
files relocated to `archive/` or deleted. See `FINAL_CHECKPOINT_CLEANUP_REPORT.md`.

## Active Checkpoints

| Directory | File | Used by | Notes |
|-----------|------|---------|-------|
| `network_v9_v3/` | `network_it_best.pt` | §3 headline table (NSL-KDD, TON_IoT, OpTC, CICIDS2017) | **Canonical publication checkpoint.** Self-supervised hybrid topo-NT-Xent + SimCLR loss on multi-network-IT data. |
| `unsw_supcon_v7/` | `best.pt` | UNSW-NB15 in `run_gate_tuning.py` (per-dataset override) | SupCon fine-tune on UNSW-NB15 campaigns. |
| `siem_supcon_v4/` | `best.pt` | SQTK_SIEM in `run_gate_tuning.py` (per-dataset override) + fixed `test_indices.npy` | SupCon fine-tune on SIEM kcluster labels. |
| `multidomain_v2/` | `best_supervised.pt` | Cross-dataset generalization study | Multi-domain backbone (UNSW + BETH + OpTC). Inference uses backbone embeddings only; supervised head discarded. |
| `multidomain_v2_optc_finetuned/` | `best_supervised.pt` | OpTC fine-tuning experiments | Domain-specific fine-tune of `multidomain_v2`. |
| `archive/` | — | Historical reference only | Legacy / failed experiment checkpoints. Not used by any active script. |

## Default Checkpoint

```python
# Canonical checkpoint for §3 headline experiments and `run_gate_tuning.py`
DEFAULT_CHECKPOINT = "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
```

`run_gate_tuning.py` overrides this default per-dataset via the
`checkpoint_override` field in `DATASET_CONFIG` (UNSW-NB15 → `unsw_supcon_v7`,
SQTK_SIEM_kcluster → `siem_supcon_v4`).

## Architecture

Single-layer (`num_layers=1`) heterogeneous GAT (`MITREHeteroGNN`), 128-dim
embeddings. Inference is fully unsupervised: embeddings → PCA whitening (16 dims)
→ optional UMAP → HDBSCAN (or Spectral for SQTK_SIEM). The cluster classifier
head is loaded but discarded at inference (`pure_unsupervised=True`).

## Known Limitations

- Backbones trained on network-traffic datasets are out-of-distribution for
  pure host-based telemetry. Use SupCon fine-tuned checkpoints
  (`unsw_supcon_v7`, `siem_supcon_v4`) for those domains.
- All checkpoints predate the v2.43 multi-seed validation; they are stable
  given fixed inputs but their *training* was single-seed (see §14 of
  `MITRE-design.md`).

## Cleanup History

- **May 2026**: Removed 7 orphaned JSON configs and stale README entries.
  Canonical checkpoint set to `network_v9_v3/network_it_best.pt`.
- **March 2026**: Retired the unstable `unsw_nb15_retrained` ARI=0.8994 claim
  (was based on a single seed and supervised head; not reproducible under
  pure-unsupervised inference).
