# NSL-KDD — Learnings (carry forward to next datasets)

Findings from the NSL-KDD lifecycle that should shape how we approach UNSW-NB15 and beyond.

---

## L1 — Protocol before tuning
The first protocol improvement (freezing the sampled subset) showed that apparent "improvements" can be noise from evaluation-set drift. On NSL-KDD, unfrozen sampling produced inflated cross-seed variance. **Freeze splits before running any baseline comparisons.**

## L2 — Standalone clustering sweeps don't transfer to the full engine
Phase 3 ran a HDBSCAN sweep on standalone cached embeddings. The winning config (mcs=10, pca=8, eps=0.0) produced catastrophic ARI degradation (0.632 → 0.078) when used with the full V3CorrelationEngine. The engine's EmbeddingConfidenceScorer post-processing changes the effective embedding geometry. **Always sweep through `clustering_sweep_full_engine.py`, never standalone HDBSCAN on raw cached embeddings.**

## L3 — attack_f1 saturates on NSL-KDD-style datasets
Binary attack/benign split + HGNN's strong ability to separate normal traffic → attack_f1 ≈ 1.0 for all methods. The metric is useless for ranking. **Check attack_f1 saturation early on each new dataset; use attack_f1_demoted to catch trivial clusterings instead.**

## L4 — campaign_id is a cleaner label track than tactic
MITRE tactic mapping merges structurally distinct attacks into coarse buckets. campaign_id (15 classes, dev ARI=0.675) > tactic (8 classes, dev ARI=0.632). **For each new dataset, audit all candidate label tracks before committing to a primary. Evaluate tactic AND a finer-grained track.**

## L5 — V3 has large cross-seed variance on NSL-KDD
V3 tactic ARI: 0.602 ± 0.079 (std is 13% of mean). The baselines are more stable. This warrants investigation on UNSW: is it inherent to the dataset, or a clustering initialization issue? **Report std alongside mean; flag if V3 std/mean ratio exceeds baseline ratio by >2×.**

## L6 — Spectral on raw features is surprisingly strong
Spectral (raw) at ARI=0.414 outperforms all embedding-based baselines except V3. This is a dataset-specific finding (NSL-KDD has relatively clean linear structure in raw features). **Do not assume raw-feature spectral is weak on new datasets; always include it in the baseline roster.**

## L7 — Silhouette (cosine) is misleading on NSL-KDD graph embeddings
V3 silhouette = −0.54 despite strong ARI=0.602. The cosine silhouette does not reflect extrinsic cluster quality on graph embeddings with this topology. **Use silhouette as diagnostic only; never as a ranking metric; note this limitation explicitly in each dataset's baseline doc.**

## L8 — The +0.19 ARI margin is the right freeze threshold
Master plan sets the gate at +0.1. NSL-KDD froze at +0.19, which felt right. **Apply the gate strictly on new datasets; a smaller margin warrants investigation before freeze, not blind acceptance.**
