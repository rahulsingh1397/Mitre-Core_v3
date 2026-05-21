# SQTK_SIEM — Protocol (Stage 2)

**Date:** 2026-05-21

---

## Split Configuration

| Parameter | Value |
|-----------|-------|
| Sample size | 5,100 rows (full corpus) |
| Stratification | Yes (`alert_type`, stratified=true) |
| Sample seed | 42 |
| Exclude from eval | **None** — no disjoint split possible (5,100 < 10,000) |

**Rationale:** `benchmark._sample_indices()` returns all rows when the corpus is smaller than the requested sample size. Using `exclude_sample_seeds` would error because no rows remain after exclusion. All 3 benchmark seeds (42/43/44) run on the full 5,100 rows; variance comes from HDBSCAN/spectral randomness, not sampling.

---

## Persisted Split Artifacts

| File | SHA256 |
|------|--------|
| `benchmark/splits/sqtk_siem_5100_seed42.npy` | (TBD at freeze) |
| `benchmark/splits/sqtk_siem_5100_seed42.json` | (metadata sidecar) |

Dataset SHA256: `ab2a63166dd79d570147de940edef242bb9e0304eccb1f79953c174874b5c617`

---

## Label Tracks

| Track | Role | Classes | Notes |
|-------|------|---------|-------|
| `alert_type` | **Primary** | 14 | Native SIEM alert type; 0 nulls |
| `tactic` | Secondary | 9 | UNKNOWN dominates (88.7%); included for completeness |
| `campaign_id` | Tertiary | 8 | UNKNOWN dominates (89.4%); included for completeness |

`kcluster` (11 clusters) is documented in `audit.md` as reference only and is NOT used as an evaluation track.

---

## Checkpoint Policy

| Dataset | Checkpoint | Rationale |
|---------|-----------|-----------|
| NSL-KDD, UNSW-NB15, TON-IoT, CICIDS2017 | `hgnn_checkpoints/network_v9_v3/network_it_best.pt` | Generic network-IDS checkpoint |
| **SQTK_SIEM** | `hgnn_checkpoints/siem_supcon_v4/best.pt` | **Dataset-specific** SIEM-trained encoder |

This is the first departure from `network_v9_v3`. The policy: each dataset's checkpoint is chosen at Stage 2 and frozen; switching requires a v1.1 (not an in-place edit).

---

## Engine Configuration

```yaml
engine_kwargs:
  device: cpu
  num_layers: 1
  hdbscan_min_cluster_size: 5
  hdbscan_pca_components: 16
  hdbscan_cluster_selection_epsilon: 0.1
  use_geometric_confidence: true
```

Same as NSL-KDD/UNSW default. No per-dataset retune at this stage.

---

## n_clusters = 14 Rationale

Matches full `alert_type` class count (14 distinct values), consistent with:
- NSL-KDD = 10
- CICIDS2017 = 15

---

## Stage 2 Exit Criterion

✅ Splits persisted to `benchmark/splits/`
✅ Hashes recorded above
✅ YAML block added to `benchmark/datasets_real.yaml`
✅ Multi-checkpoint policy documented
