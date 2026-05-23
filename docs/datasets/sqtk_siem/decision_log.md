# SQTK_SIEM — Decision Log

**Date:** 2026-05-21

---

## 2026-05-21 — No disjoint dev/eval split

- **Choice:** Single YAML entry with `sample_size: 5100`, no `exclude_sample_seeds`.
- **Alternative considered:** Force a disjoint split by using `sample_size: 2500` for dev and eval.
- **Rationale:** 5,100 rows < 10,000 sample size. `_sample_indices()` returns all rows when corpus < sample. Using `exclude_sample_seeds` would raise "Not enough rows remaining" error. A 2,500-row subset would discard >50% of the data on an already-small dataset.
- **Impact:** All 3 benchmark seeds run on identical indices; variance comes from clustering randomness only.

## 2026-05-21 — Dataset-specific checkpoint

- **Choice:** Use `siem_supcon_v4/best.pt` instead of `network_v9_v3`.
- **Alternative considered:** Stay on `network_v9_v3` for consistency.
- **Rationale:** SQTK_SIEM is a SIEM dataset with hostname/username fields that `network_v9_v3` was not trained on. The SIEM-specific checkpoint may better exploit heterogeneous graph structure.
- **Impact:** First multi-checkpoint policy in benchmark. Documented in `protocol.md`.

## 2026-05-21 — n_clusters = 14

- **Choice:** Set `n_clusters: 14` to match full `alert_type` class count.
- **Alternative considered:** 11 (matching kcluster count).
- **Rationale:** kcluster is pre-computed from another algorithm and must not be used as a target. 14 is the ground-truth alert_type count, consistent with NSL-KDD=10, CICIDS2017=15.

## 2026-05-21 — tactic/campaign_id as secondary tracks despite UNKNOWN dominance

- **Choice:** Include `tactic` and `campaign_id` as `alt_label_cols` despite 88–89% UNKNOWN.
- **Alternative considered:** Exclude them (low expected ARI).
- **Rationale:** Lifecycle protocol requires all candidate tracks to be evaluated for completeness. Their low ARI is itself a dataset characteristic worth documenting.

## 2026-05-21 — kcluster excluded from evaluation

- **Choice:** Do NOT use `kcluster` as a label track.
- **Alternative considered:** Include as a comparison track.
- **Rationale:** kcluster is a pre-computed clustering result (11 clusters). Using it as ground truth would be circular — comparing V3 against another clustering algorithm's output.

## 2026-05-21 — Stage 4 Decision Gate: Path B

- **Choice:** Document Path B (V3 loses) and freeze v1.0 without retuning.
- **Alternative considered:** Immediate retuning via `clustering_sweep_full_engine.py`.
- **Rationale:** V3 ARI = 0.3551 vs best baseline PCA + HDBSCAN ARI = 0.3825. Margin = -0.027 (< 0.1). Per lifecycle protocol, write `investigation.md` before ANY retuning. ~~The root cause is embedding collapse (over-smoothing, mean cosine similarity = 0.958) from the `siem_supcon_v4` checkpoint.~~ **[CORRECTION 2026-05-23: the cosine_sim=0.958 figure was a measurement artifact — `alert_feature_dim` was hardcoded to 6 in the diagnostic script while the checkpoint expects 15-dim. Re-measured value = 0.79. Root cause re-opened; investigation.md is invalidated. See `docs/experiments/multi_layer_depth.md` for corrected analysis.]**
- **Impact:** `investigation.md` written. No retuning initiated. v1.0 freeze captures the honest result.
