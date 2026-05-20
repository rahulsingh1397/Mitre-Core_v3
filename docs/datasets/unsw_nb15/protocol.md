# UNSW-NB15 — Protocol (Stage 2)

**Date:** 2026-05-19

---

## Split Configuration

| Parameter | Value |
|-----------|-------|
| Sample size | 10,000 rows |
| Stratification | Yes (`attack_cat`, stratified=true) |
| Dev seed | 42 |
| Eval seed | 142 |
| Exclude from eval | dev indices (seed 42) — disjoint enforced |

**Rationale for 10K (not 20K):** Matches NSL-KDD protocol exactly for cross-dataset ARI comparability. UNSW has 175,341 rows (>NSL-KDD's 125,973), so a 10K sample is proportionally comparable. The Worms class (130 rows) is the binding constraint — raising to 20K roughly doubles Worms representation from ~7 to ~14 rows, which is still too sparse for reliable cluster recall. The known limitation is documented in `audit.md` and will be noted in `v1.0_baseline.md`.

---

## Persisted Split Artifacts

| File | SHA256 |
|------|--------|
| `benchmark/splits/unsw_nb15_10000_seed42.npy` | `f17141d3a5380f595b6613efac4129bb70efeb22f75928f9ded7a6a0854a9dc7` |
| `benchmark/splits/unsw_nb15_10000_seed42.json` | (metadata sidecar) |
| `benchmark/splits/unsw_nb15_10000_seed142.npy` | `b137b0700bf7bd267447f302be173b5d967366d93588446e680ef5cc9444bff2` |
| `benchmark/splits/unsw_nb15_10000_seed142.json` | (metadata sidecar) |

Dataset SHA256: `c7856d8428fd7b35ffd233ccece378be3e0b2ba9d23c6b7bfe37dab13441b892`

---

## Label Tracks

| Track | Role | Classes | Notes |
|-------|------|---------|-------|
| `attack_cat` | **Primary** | 10 | Native UNSW label; 0 nulls |
| `tactic` | Secondary | 8 (after NaN→"Normal" fill) | Mirrors NSL-KDD primary for cross-dataset comparison |
| `alert_type` | Binary | 2 | For binary_ari headline |

---

## Engine Configuration (matching NSL-KDD default)

```yaml
engine_kwargs:
  device: cpu
  num_layers: 1
  hdbscan_min_cluster_size: 5
  hdbscan_pca_components: 16
  hdbscan_cluster_selection_epsilon: 0.1
  use_geometric_confidence: true
```

Same as NSL-KDD frozen config. No per-dataset retune at this stage (per master plan Stage 2 spec).

---

## Disjoint Verification

Dev (seed 42) and eval (seed 142) index sets have zero overlap — verified programmatically. Both splits reference the same dataset SHA256, confirming they were generated from the same file.

---

## Stage 2 Exit Criterion

✅ Splits persisted to `benchmark/splits/`
✅ Splits verified disjoint (0 overlap)
✅ Hashes recorded above
✅ YAML blocks added to `benchmark/datasets_real.yaml` (UNSW-NB15-dev disabled; UNSW-NB15 enabled)
