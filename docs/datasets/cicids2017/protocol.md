# CICIDS2017 — Protocol

## Split Configuration

| Parameter | Dev | Eval |
|---|---|---|
| Dataset name | CICIDS2017-dev | CICIDS2017 |
| sample_size | 10,000 | 10,000 |
| sample_seed | 42 | 142 |
| exclude_sample_seeds | — | [42] |
| stratified_sample | true | true |
| split_group | cicids2017 | cicids2017 |
| Seeds (benchmark) | 42, 43, 44 | 42, 43, 44 |

Dev and eval splits are guaranteed disjoint by the `exclude_sample_seeds: [42]` mechanism —
the eval split materializtion excludes all indices from the dev (seed=42) split before sampling.

## Label Configuration

| Parameter | Value |
|---|---|
| label_col | alert_type |
| alt_label_cols | tactic, campaign_id |
| n_clusters | 15 |
| Checkpoint | hgnn_checkpoints/network_v9_v3/network_it_best.pt |

**n_clusters=15 rationale:** The full corpus has 15 named `alert_type` classes. K-Means receives
the full-schema count as its privileged prior, consistent with NSL-KDD (10), UNSW-NB15 (10), and
TON_IoT (10). The 3 absent classes in the 10K subset result in 3 empty clusters — expected
behavior, not a bug.

## Null-Row Handling

288,602 rows (9.25%) have null `alert_type`, `timestamp`, `src_ip`, and `dst_ip`. These rows
receive class label `"UNKNOWN"` via `fillna("UNKNOWN")` in the benchmark's stratified sampler.
Approximately 925 "UNKNOWN" rows appear in each 10K sample. They become isolated alert nodes in
the graph (no IP edges due to null src_ip/dst_ip) and contribute a 13th class to alert_type
evaluation. This is the correct behavior — exclusion would require patching the benchmark loader
and would deviate from the lifecycle established for prior datasets.

## Engine Configuration

Same as TON_IoT (no per-dataset retune at Stage 2):

```yaml
engine_kwargs:
  device: cpu
  num_layers: 1
  hdbscan_min_cluster_size: 5
  hdbscan_pca_components: 16
  hdbscan_cluster_selection_epsilon: 0.1
  use_geometric_confidence: true
```

## Split Files

| File | Description |
|---|---|
| `benchmark/splits/cicids2017_10000_seed42.npy` | Dev split indices |
| `benchmark/splits/cicids2017_10000_seed42.json` | Dev split metadata |
| `benchmark/splits/cicids2017_10000_seed142.npy` | Eval split indices |
| `benchmark/splits/cicids2017_10000_seed142.json` | Eval split metadata |

## Decision Log References

- `decision_log.md` — records the n_clusters=15 choice and null-row handling decision
