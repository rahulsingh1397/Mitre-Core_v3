# DARPA OpTC — Benchmark Protocol

## Checkpoint Decision

After Read A of `hgnn/hgnn_correlation.py`, we confirmed that `use_geometric_confidence` is a runtime flag in `engine_kwargs`. When `True`, the engine uses GAEC/HDBSCAN on backbone embeddings before the classification head. When `False`, it uses softmax on `cluster_logits`. The checkpoint type does **not** bypass this flag.

**Decision**: GAEC overrides -> use `hgnn_checkpoints/multidomain_v2_optc_finetuned/best_supervised.pt`.

Rationale: The `multidomain_v2_optc_finetuned` checkpoint was explicitly fine-tuned on OpTC data (val_ari peaked at 0.818). Since GAEC uses the backbone embeddings and ignores the classification head, this checkpoint's learned graph representations should be optimal for OpTC.

## Preprocessor Decision

After Read B of `datasets/loaders/darpa_optc_loader.py`, we confirmed that the loader renames raw OpTC columns to MITRE-standard names (`SourceAddress`, `DestinationAddress`, `SourceHostName`, `SourceUserName`, `CampaignId`, `Tactic`). The existing `processed_optc_full.csv` already contains these renamed columns.

**Decision**: No preprocessor needed. Use `datasets/DARPA_OpTC/processed_optc_full.csv` directly.

## Key Protocol Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_clusters` | 2 | Binary dataset: exactly 2 campaigns (Benign, RedTeam_Sep23) |
| `sample_size` | 10000 | Standard benchmark size; 4.65M >> 10K so disjoint dev/eval split works normally |
| `stratified_sample` | true | Ensures ~420 RedTeam rows in each 10K split (4.2% of 10,000) |
| `clustering_method` | hdbscan | GAEC uses HDBSCAN on backbone embeddings |
| `label_col` | CampaignId | Binary campaign classification |
| `alt_label_cols` | Tactic | Secondary track (degenerate — only Execution value) |

## Temporal Leakage Mitigation

Attacks in OpTC are confined to 2019-09-23/25. Without stratification, a random 10K sample might miss all RedTeam rows. Stratified sampling by `campaign_id` ensures proportional representation: ~420 RedTeam rows per 10K split, preventing both leakage and class imbalance bias.

## Binary Metric Rationale

**Headline metric: binary_ARI** (not standard ARI).

Standard ARI is structurally low on 2-class datasets because fine-grained sub-clustering within the two true campaigns produces chance-level ARI by design. `binary_ari` maps each predicted cluster to its majority ground-truth label (Benign vs RedTeam) before computing ARI, making it the meaningful comparison metric for OpTC.

## Supervised Reference

The `multidomain_v2_optc_finetuned` checkpoint achieved val_ari = 0.818 during supervised training. This is noted as a supervised reference only — it is **not** part of the unsupervised benchmark roster.
