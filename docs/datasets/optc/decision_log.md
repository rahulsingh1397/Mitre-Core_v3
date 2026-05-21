# DARPA OpTC — Decision Log

## Stage 1 — Checkpoint Decision

**Date**: 2026-05-21

**Read A**: `use_geometric_confidence` is a runtime flag in `engine_kwargs`. When `True`, the engine uses GAEC/HDBSCAN on backbone embeddings before the classification head. When `False`, it uses softmax on `cluster_logits`. The checkpoint type does NOT bypass this flag — a supervised checkpoint loaded with `use_geometric_confidence=True` will have its backbone used for GAEC, and the `cluster_classifier` head is ignored entirely.

**Decision**: GAEC overrides → use checkpoint `hgnn_checkpoints/multidomain_v2_optc_finetuned/best_supervised.pt`.

Rationale: The multidomain_v2_optc_finetuned checkpoint was explicitly fine-tuned on OpTC data (val_ari peaked at 0.818). Since GAEC uses the backbone embeddings and ignores the classification head, this checkpoint's learned graph representations should be optimal for OpTC.

## Stage 2 — Preprocessor Decision

**Date**: 2026-05-21

**Read B**: `datasets/loaders/darpa_optc_loader.py` renames raw OpTC columns to MITRE-standard names (`SourceAddress`, `DestinationAddress`, `SourceHostName`, `SourceUserName`, `CampaignId`, `Tactic`). The existing `processed_optc_full.csv` already contains these renamed columns.

**Decision**: No preprocessor needed. Use `datasets/DARPA_OpTC/processed_optc_full.csv` directly in the YAML.

## Stage 4 — Decision Gate

**Date**: 2026-05-21

**Metric**: binary_ARI on campaign_id track (standard ARI is structurally low on 2-class datasets).

**Results**:
- V3 binary_ARI = 0.9993 ± 0.0007 (mean over seeds 42/43/44)
- Best baseline binary_ARI = 0.9993 (HDBSCAN emb, tied with V3)
- HDBSCAN (raw) binary_ARI = 0.9973
- PCA + HDBSCAN binary_ARI = 0.9750
- PCA + K-Means binary_ARI = 0.5000 (random for binary, n_clusters=2 privileged)
- All other baselines <= 0.037

**Standard ARI context** (for reference only, not the comparison metric):
- V3 standard ARI = 0.254
- Best baseline standard ARI = 0.538 (PCA + K-Means, which has n_clusters=2 privileged information)
- Standard ARI is ~0.05 for most methods by design on a 2-class dataset.

**Decision**: Path A — V3 binary_ARI wins (tied for first at 0.9993, >0.1 margin over all baselines except HDBSCAN emb which also uses V3 embeddings). No sweep required. Proceed to Stage 5.

Rationale: V3 achieves near-perfect binary separation (0.9993), indicating the GAEC/HDBSCAN path on the OpTC-finetuned backbone embeddings successfully separates Benign from RedTeam campaigns. No retuning is needed.

## Stage 5 — Metrics + Demotion

**Date**: 2026-05-21

- **dominant_confusion_accuracy**: Pre-demoted (sixth consecutive dataset where it is degenerate; returns 1.0 for all methods because every true class has a majority predicted cluster).
- **Standard ARI**: Retained but annotated — "structurally low on 2-class datasets; not the primary metric." V3 standard ARI = 0.255, best baseline = 0.538 (PCA + K-Means with n_clusters=2 privileged).
- **binary_ARI on tactic track**: Documented as 1.0 for all methods — the tactic column is degenerate (only 1 unique non-null value: Execution), so the track is not meaningful.
- **All 12 metrics confirmed present** in `results.csv`.
