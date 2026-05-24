# V3 Architecture Improvement Experiments

## Ablation Record

The cumulative ablation table for all Part X experiments lives at
[`docs/ablations/v3_ablation_record.md`](../ablations/v3_ablation_record.md)
(human view) and [`docs/ablations/v3_ablation_record.csv`](../ablations/v3_ablation_record.csv)
(machine view). Every experiment in this directory contributes rows to that record.

## Failure Mode Map

| Dataset | V3 ARI | Failure Diagnosed | Fix Target |
|---------|--------|-------------------|------------|
| TON_IoT | 0.423 (loses by 0.199) | HDBSCAN over-segmentation (55 clusters for 10) | Hard negative mining + deeper network |
| CICIDS2017 | 0.177 (loses by 0.156) | 72.9% BENIGN → 30+ sub-clusters | Hard negative mining + imbalance-aware positives |
| SQTK_SIEM | 0.355 (loses by 0.027) | Embedding collapse: cosine_sim=0.958 | Multi-layer + stronger loss |
| NSL-KDD | 0.602 (wins) | Ceiling analysis | HGT architecture |
| UNSW-NB15 | 0.564 (wins) | Ceiling analysis | HGT architecture |
| DARPA OpTC | 0.999 binary_ARI | Near-perfect | N/A |

## Experiments (ordered by effort)

### Exp 1 — Hard Negative Mining
- **Status**: Ready to train
- **Script**: `training/train_v9_hard_negatives.py`
- **Checkpoint**: `hgnn_checkpoints/network_v9_v4_hardneg/`
- **Change**: Replace random negatives with 75% semi-hard / 25% random
- **Target**: CICIDS2017 ARI > 0.25, TON_IoT ARI > 0.50
- **Effort**: ~4h retrain, no architecture change
- **Run**: `python training/train_v9_hard_negatives.py --epochs 150`

### Exp 2 — Multi-layer Depth + Collapse Diagnostic
- **Status**: Diagnostic ready (no retrain for 2a)
- **Script**: `scripts/measure_embedding_collapse.py`
- **Change**: Test num_layers=2/3 at inference to diagnose SQTK_SIEM collapse
- **Target**: cosine_sim < 0.90, SQTK_SIEM ARI > 0.382
- **Effort**: 2a: no retrain; 2b: ~2h if needed
- **Run**: `python scripts/measure_embedding_collapse.py --all`

### Exp 3 — Feature Dimension Ablation (6→15 dims)
- **Status**: Ready to train
- **Script**: `training/train_v9_15dim.py`
- **Checkpoint**: `hgnn_checkpoints/network_v9_15dim/`
- **Change**: Enable 9 contextual features (ports, bytes, burstiness) already computed by data pipeline
- **Target**: ≥2 datasets improve ARI by > 0.05
- **Effort**: ~4h retrain
- **Run**: `python training/train_v9_15dim.py --epochs 150`

### Exp 4 — HGT Architecture
- **Status**: Ready to train
- **Model**: `hgnn/hgnn_correlation_hgt.py` → `MITREHeteroGNN_HGT`
- **Script**: `training/train_v9_hgt.py`
- **Checkpoint**: `hgnn_checkpoints/network_v9_hgt/`
- **Change**: Replace GATConv + HeteroConv with PyG HGTConv (type-specific attention)
- **Target**: ≥3/6 datasets improve ARI
- **Effort**: ~6h retrain
- **Run**: `python training/train_v9_hgt.py --epochs 150 --num_layers 2`

### Exp 5 — Positional Encoding (conditional on Exp 4)
- **Status**: Implementation ready, deferred
- **Module**: `mitre_core/graph/positional_encoding.py`
- **Change**: Het-node2vec type-aware random walks → 16-dim structural PE
- **Target**: Only if Exp 4 shows IP-only graph benefit
- **Effort**: ~3h retrain
- **Dependency**: Exp 4

## Key Policies

1. **v1.0 frozen artifacts are never touched** — improvements produce v1.1/ directories alongside
2. Each experiment gets documentation in this directory
3. **Decision gate** at each step: if no improvement, document and move on
4. All checkpoints saved with `experiment` key for traceability

## Execution Order

```
Exp 1 (hard neg) ──→ benchmark ──→ decision gate
                                        │
Exp 2 (collapse) ──→ diagnostic ───────→│
                                        ↓
                                   Exp 3 (15dim) ──→ benchmark ──→ decision gate
                                                                        │
                                                                        ↓
                                                                   Exp 4 (HGT) ──→ benchmark
                                                                                        │
                                                                                        ↓
                                                                                   Exp 5 (PE, conditional)
```
