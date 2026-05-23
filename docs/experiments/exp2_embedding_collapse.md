# Exp 2 — Multi-layer Depth & Embedding Collapse Diagnostic

## Hypothesis
SQTK_SIEM's cosine_sim=0.958 indicates severe embedding collapse — all alert
embeddings converge to nearly the same point on the unit hypersphere. With 1 GNN
layer, the model lacks representational depth to separate structurally distinct
alerts.

Increasing `num_layers` from 1→2/3 expands the receptive field and may break
the collapse by enabling multi-hop structural information.

## Change Summary
- **Script**: `scripts/measure_embedding_collapse.py`
- **Phase 2a**: Diagnostic only — test num_layers=1/2/3 at inference (no retrain)
- **Phase 2b**: If collapse breaks at deeper layers, retrain with num_layers=2
- **Metrics**: Mean pairwise cosine similarity, participation ratio, uniformity

## Run Commands

### 2a: Diagnostic (no GPU needed)
```bash
# Single dataset
python scripts/measure_embedding_collapse.py --dataset sqtk_siem

# All datasets
python scripts/measure_embedding_collapse.py --all

# Specific layer count
python scripts/measure_embedding_collapse.py --dataset sqtk_siem --num_layers 2
```

### 2b: Retrain (if collapse breaks at deeper layers)
```bash
python training/train_graph_mae_v9_multidata_fast.py \
    --epochs 150 \
    --output_dir ./hgnn_checkpoints/network_v9_v3_deep
# Note: Would need to modify the script to accept --num_layers argument
```

## Output
Results saved to `experiments/results/exp2_collapse_<dataset>.json` with:
- `mean_cosine_sim`: Target < 0.90
- `participation_ratio`: Effective dimensionality
- `uniformity`: Wang & Isola 2020 metric

## Success Criteria
- cosine_sim < 0.90 at some layer depth
- SQTK_SIEM ARI > 0.382 (baseline: 0.355)

## Decision Gate
- If deeper layers break collapse → retrain with optimal depth (2b)
- If collapse persists at all depths → collapse is in the encoder, not message passing
  → consider stronger loss (triplet margin loss or VICReg)

## Results (2026-05-23 — Run Complete)
**Three script bugs were fixed before the successful run** (see commit history):
1. Removed premature dummy forward pass that materialized lazy layers to wrong feature dim
2. Added dynamic feature padding/truncation to match encoder expectations
3. Fixed `alert_feature_dim` from hardcoded 6 → detected from `alert_raw_proj.weight.shape[1]`
Bug #3 was the critical one: `siem_supcon_v4` expects 15-dim inputs, not 6. The mismatch
produced uniformly garbled embeddings with cosine_sim ≈ 0.96 — the original "collapse" diagnosis
was entirely caused by this bug.
| Dataset | L=1 | L=2 | L=3 | Status | v1.0 ARI |
|---|---|---|---|---|---|
| sqtk_siem | 0.79 | 0.79 | 0.79 | ✅ OK | 0.355 (loses) |
| cicids2017 | 0.60 | 0.60 | 0.61 | ✅ OK | 0.177 (loses) |
| ton_iot | 0.73 | 0.73 | 0.74 | ✅ OK | 0.423 (loses) |
| unsw_nb15 | 0.74 | 0.77 | 0.75 | ✅ OK | 0.564 (wins) |
| nsl_kdd | 0.70 | 0.70 | 0.71 | ✅ OK | 0.602 (wins) |
Threshold for "collapsed": cosine_sim > 0.90. **All 15 measurements pass — no dataset is collapsed.**
**Key findings:**
- Exp 2 hypothesis REJECTED: no embedding collapse exists anywhere
- SQTK_SIEM cosine_sim = 0.79, not 0.958 (measurement artifact from script bug)
- Multi-layer depth changes cosine_sim by at most ±0.03 — confirms single-layer is fine
- Inverse correlation: lower cosine_sim ↔ worse ARI on losing datasets (CICIDS2017: 0.60, worst loss)
- Bottleneck is HDBSCAN density-finding, not embedding quality
**Artifacts:** `experiments/results/exp2_collapse_all.json`, `experiments/results/exp2_collapse_log.txt`
**Next experiments:** Exp 2.5 (HDBSCAN → GMM+BIC, 30 min CPU), Exp 2.6 (PCA preprocessing parity, 15 min CPU)
