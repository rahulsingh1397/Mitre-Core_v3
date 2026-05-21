# SQTK_SIEM — Investigation (Stage 4, Path B)

**Date:** 2026-05-21
**Path:** B — V3 does NOT win by >0.1 ARI margin (margin = -0.027 vs best baseline)

---

## Decision Gate Results

| Method | ARI mean ± std (alert_type) | AMI mean ± std (alert_type) |
|---|---|---|
| **MITRE-CORE V3** | 0.3551 ± 0.0000 | 0.6039 ± 0.0000 |
| PCA + HDBSCAN | **0.3825 ± 0.0000** | 0.4941 ± 0.0000 |
| PCA + K-Means | 0.3686 ± 0.1435 | 0.5762 ± 0.1211 |

**Best baseline:** PCA + HDBSCAN (ARI = 0.3825)  
**V3 margin:** -0.027 (loses)  
**Path taken:** B — investigation before retuning.

---

## Hypothesis 1: Embedding Collapse (Over-smoothing)

**Evidence:** During V3 inference, the following warning was emitted:

> OVER-SMOOTHING DETECTED: mean pairwise cosine similarity=0.9577 > 0.95. Embeddings have collapsed. Consider reducing num_layers to 1 or adding residual skip connections to MITREHeteroGNN.

**Interpretation:** The `siem_supcon_v4/best.pt` checkpoint produces near-identical embeddings for most alerts (cosine similarity 0.96). This collapses the cluster structure that HDBSCAN depends on.

**Why this matters:**
- HDBSCAN relies on local density gradients in embedding space.
- If all embeddings are nearly identical, density becomes uniform → HDBSCAN either merges everything or fragments randomly.
- V3 produces 42 clusters (alert_type has 14 classes), suggesting fragmentation of the collapsed space.

**Why PCA + HDBSCAN wins:**
- PCA + HDBSCAN operates on the raw tabular features (18 columns) which are NOT collapsed.
- Raw features preserve the actual alert structure (src_ip, dst_ip, alert_type names encoded, etc.).
- The SIEM-specific encoder has learned to suppress feature variance — possibly due to SupCon training pushing same-class alerts together and different-class alerts apart, but on this dataset the margin is too large and everything collapses to a single region.

---

## Hypothesis 2: SupCon Training Regime Mismatch

**Observation:** `siem_supcon_v4` was trained with supervised contrastive loss. SupCon optimizes for class-separability in embedding space. On datasets with many fine-grained classes (14 alert types), SupCon may over-optimize, collapsing all within-class variation to near-zero.

**Contrast with network_v9_v3:**
- `network_v9_v3` was trained with GraphMAE (generative SSL), which preserves more feature variance.
- On NSL-KDD/UNSW/CICIDS2017, `network_v9_v3` outperforms baselines.
- On SQTK_SIEM, the SIEM-specific SupCon checkpoint underperforms the generic GraphMAE checkpoint would likely achieve.

---

## Hypothesis 3: Small Dataset + Deep Encoder = Overfitting Artifact

**Observation:** SQTK_SIEM has only 5,100 rows. The `siem_supcon_v4` checkpoint may have been trained on a much larger SIEM corpus and memorized patterns that don't generalize to this specific 5,100-row snapshot.

**Supporting evidence:**
- V3 ARI is identical across all 3 seeds (0.35507) — the model output is completely deterministic, but the clustering quality is poor.
- This suggests the model is not "noisy" — it's consistently producing the same bad embeddings.

---

## Recommended Actions (before any retuning)

1. **Test generic checkpoint:** Run `network_v9_v3/network_it_best.pt` on SQTK_SIEM to verify whether the issue is checkpoint-specific or dataset-inherent.
2. **Embedding visualization:** Plot t-SNE of V3 embeddings vs raw features to confirm collapse visually.
3. **num_layers sweep:** The warning explicitly suggests "reducing num_layers to 1." Current config already uses `num_layers: 1`, so this is not the fix.
4. **hdbscan_min_cluster_size sweep:** V3 produces 42 clusters on 14 classes. A larger `min_cluster_size` might force merging. But given embedding collapse, this is treating the symptom.

---

## Stage 4 Exit Criterion

Path B documented. No retuning initiated until root cause is verified.
