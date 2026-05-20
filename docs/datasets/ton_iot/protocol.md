# TON_IoT — Protocol (Stage 2)

**Date:** 2026-05-20

---

## Split Configuration

| Parameter | Value |
|-----------|-------|
| Sample size | 10,000 rows |
| Stratification | Yes (`alert_type`, stratified=true) |
| Dev seed | 42 |
| Eval seed | 142 |
| Exclude from eval | dev indices (seed 42) — disjoint enforced |

---

## Persisted Split Artifacts

| File | SHA256 |
|------|--------|
| `benchmark/splits/ton_iot_10000_seed42.npy` | `98f63520c3f0eacec3e1da260e43a11ec99c42de83ae9ccb60b9d927c053c1f5` |
| `benchmark/splits/ton_iot_10000_seed42.json` | (metadata sidecar) |
| `benchmark/splits/ton_iot_10000_seed142.npy` | `d1819680cc56cdc28f6ae33076df75da6961c78bda79df065ed28b21ce7f7d6b` |
| `benchmark/splits/ton_iot_10000_seed142.json` | (metadata sidecar) |

Dataset SHA256: `0d307cb86b64099efb13088d94096a7863f3b5500396887eab437fb88ca0ce6f`

---

## Label Tracks

| Track | Role | Classes | Notes |
|-------|------|---------|-------|
| `alert_type` | **Primary** | 10 | Native TON_IoT attack type; 0 nulls |
| `tactic` | Secondary | 7 (after NaN→"Normal" fill) | Coarser; multiple alert_types per tactic |

Binary track: `alert_type="normal"` vs attack — handled via `benign_label: normal` in YAML if needed; the standard multi-class ARI already encodes this.

---

## Engine Configuration (matching NSL-KDD/UNSW default)

```yaml
engine_kwargs:
  device: cpu
  num_layers: 1
  hdbscan_min_cluster_size: 5
  hdbscan_pca_components: 16
  hdbscan_cluster_selection_epsilon: 0.1
  use_geometric_confidence: true
```

Same as NSL-KDD/UNSW frozen config. No per-dataset retune at this stage.

---

## Schema Notes

TON_IoT parquet has `src_port`, `dst_port`, `label` (extra) and lacks `hostname`, `username`, `src_bytes`, `dst_bytes`, `stage`, `service`. The graph converter handles missing hostname/username via `dropna()` fallback. V3 smoke-tested on 500 rows: 29 clusters, 0% noise.

---

## Disjoint Verification

Dev (seed 42) and eval (seed 142) index sets have zero overlap — verified. Both reference dataset SHA256 above.

---

## Stage 2 Exit Criterion

✅ Splits persisted to `benchmark/splits/`
✅ Splits verified disjoint (0 overlap)
✅ Hashes recorded above
✅ YAML blocks added to `benchmark/datasets_real.yaml` (TON-IoT-dev disabled; TON-IoT enabled)
