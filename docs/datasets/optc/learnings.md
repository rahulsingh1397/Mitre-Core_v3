# DARPA OpTC — Learnings

> Filled at Stage 6 after v1.0 freeze.

## What Worked

1. **Checkpoint choice was correct**: The `multidomain_v2_optc_finetuned` backbone produced near-perfect binary separation (binary_ARI = 0.999). GAEC successfully ignored the supervised classification head and used only the graph embeddings.
2. **No preprocessor needed**: The `darpa_optc_loader.py` already outputs MITRE-standard column names, so `processed_optc_full.csv` was ready for direct YAML reference.
3. **Stratified sampling is critical**: With only 4.2% RedTeam rows, random sampling could easily miss all attacks in a 10K split. Stratification guarantees ~420 RedTeam rows per split.
4. **HDBSCAN(emb) tied V3**: This validates that the V3 embeddings — not the clustering algorithm — are the primary driver of performance on OpTC.

## Surprises

1. **Standard ARI is not just low — it's structurally meaningless on 2-class**: Even the best baseline (PCA+K-Means with n_clusters=2 privileged) only reaches ARI=0.538. Most methods are at ~0.05. This confirms binary_ARI must be the headline metric.
2. **HDBSCAN(raw) achieves 0.997 binary_ARI**: The raw OpTC features alone have extremely strong class separation. The graph neural network adds only marginal value on this dataset.
3. **Tactic track is completely degenerate**: Only 1 non-null tactic value (Execution). The tactic alt_label track is not meaningful for comparison.

## Technical Notes

- The 4.6M row CSV reads successfully but takes noticeable time per seed (~8 seconds per 10K sample just for loading and feature encoding).
- `all_points_membership_vectors()` warning appears but falls back to `clusterer.probabilities_` with no impact on results.
- Temporal parsing warnings (`Could not infer format`) are cosmetic — benchmark.py handles the mixed timestamp formats correctly.

## For Future Work

- OpTC is an easy dataset for V3 (binary_ARI = 0.999). The real challenge is maintaining this performance while improving on harder datasets like CICIDS2017 and SQTK_SIEM.
- Consider using OpTC as a sanity-check dataset: if V3 binary_ARI drops below 0.99 on OpTC, something is broken in the engine or checkpoint loading.

