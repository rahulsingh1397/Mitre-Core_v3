# NSL-KDD — Benchmark Protocol

## Dataset
- Source file: `datasets/nsl_kdd/mitre_format.csv` (125,973 rows)
- SHA256: `29d0919999e2a1cc952a5cc6826b0906d3b5862108f68dc151e739415e4bb706`

## Splits
| Split | Seed | Size | File | Purpose |
|---|---|---|---|---|
| Dev | 42 | 10,000 | `benchmark/splits/nsl_kdd_10000_seed42.npy` | Tuning, investigation, sweep runs |
| Eval | 142 | 10,000 | `benchmark/splits/nsl_kdd_10000_seed142.npy` | Reported headline numbers |

Splits are stratified by `tactic`, disjoint, and frozen. The eval split must never be used for tuning.

## Sampling
- Method: stratified shuffle split
- Sample size: 10,000
- `sample_seed` is fixed (not the benchmark seed), ensuring the sampled subset is identical across all three benchmark seeds.

## Benchmark seeds
42, 43, 44 — all methods run under each seed.

## Label tracks
| Track | Column | Classes | Role |
|---|---|---|---|
| Tactic | `tactic` | 8 | Primary headline |
| Alert type | `alert_type` | 2 (binary) | Secondary (dev only in v1.0) |
| Campaign ID | `campaign_id` | 15 | Secondary (dev only in v1.0) |

## Checkpoint
`hgnn_checkpoints/network_v9_v3/network_it_best.pt`
SHA256: `c74da9b1ca6f1b7439d7a71480d0cd5da3a900df7178eb4587694734dfc48b4e`

## Engine kwargs (locked at v1.0)
```yaml
num_layers: 1
hdbscan_min_cluster_size: 5
hdbscan_pca_components: 16
hdbscan_cluster_selection_epsilon: 0.1
use_geometric_confidence: true
device: cpu
```
