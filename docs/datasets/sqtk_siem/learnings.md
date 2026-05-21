# SQTK_SIEM — Learnings

**Date:** 2026-05-21
**Status:** FROZEN — v1.0

---

## What Surprised Us

- **V3 underperforms baselines** on the dataset where heterogeneous graph structure is richest. SQTK_SIEM has hostname, username, src_ip, dst_ip — the most graph-rich dataset — yet V3 ARI (0.355) loses to PCA + HDBSCAN on raw features (0.382).
- **Embedding collapse:** The `siem_supcon_v4` checkpoint produces over-smoothed embeddings (mean pairwise cosine similarity = 0.958). SupCon training appears to have collapsed within-class variance too aggressively.
- **Deterministic V3 results across seeds:** Because the dataset is small (5,100 rows) and V3 uses no sampling randomness, ARI/AMI are identical across all 3 benchmark seeds. The only variance comes from baseline methods that use random_state (K-Means, Spectral, PCA).
- **No disjoint split needed:** When corpus < sample_size, `_sample_indices()` returns all rows. This is a clean protocol deviation — no special-case code needed.

## What Didn't Transfer from Prior Datasets

- **Multi-checkpoint policy:** All prior datasets used `network_v9_v3`. SQTK_SIEM is the first dataset-specific checkpoint, and it underperforms. This challenges the assumption that domain-specific checkpoints always help.
- **No BENIGN class:** SQTK_SIEM has no explicit benign/normal label. `attack_f1` defaults to 1.0 for all methods. This makes `attack_f1` and `binary_ari` degenerate on this dataset.
- **kcluster as reference only:** Unlike other datasets where we derive label tracks from the data, SQTK_SIEM comes with a pre-computed clustering. Using it as ground truth would be circular.

## What to Carry to the Next Dataset

1. **Read alert_feature_dim from checkpoint:** The fix in `hgnn_correlation.py` (detecting `alert_feature_dim` from checkpoint shape) should be preserved for all future checkpoint loading.
2. **Honest reporting of underperformance:** Path B (V3 loses) is a valid outcome. Freezing the honest result is better than hiding it.
3. **Embedding diagnostics:** The over-smoothing warning should be treated as a first-class diagnostic, not just a log message.
4. **Test pattern for no-split datasets:** `test_single_split_without_exclusion_covers_full_corpus` can be reused for any future dataset that is smaller than the sample size.

## Checkpoint-Specific Notes

- **siem_supcon_v4/best.pt:** Produces collapsed embeddings on SQTK_SIEM. ARI = 0.355, losing to PCA + HDBSCAN (0.382).
- **Hypothesis:** SupCon training on a larger SIEM corpus may have overfitted to that corpus's class structure, failing to generalize to the 5,100-row SQTK_SIEM snapshot.
- **Recommended follow-up:** Test `network_v9_v3/network_it_best.pt` on SQTK_SIEM to determine whether the issue is checkpoint-specific or dataset-inherent.
