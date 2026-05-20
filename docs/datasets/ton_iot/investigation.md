# TON_IoT — Stage 4 Investigation (Path B)

**Date:** 2026-05-20
**Triggered by:** V3 ARI=0.423 vs K-Means (raw) ARI=0.622 on alert_type track — margin=-0.199.

---

## Diagnostic Results

| Method | ARI | AMI | n_pred_clusters | noise_frac |
|--------|-----|-----|-----------------|------------|
| K-Means (raw) | 0.622 | 0.769 | **10** | 0.00 |
| K-Means (emb) | 0.612 | 0.784 | **10** | 0.00 |
| PCA + K-Means | 0.507 | 0.728 | **10** | 0.00 |
| **MITRE-CORE V3** | **0.423** | **0.705** | **55** | 0.00 |
| PCA + HDBSCAN | 0.248 | 0.621 | 185 | 0.06 |

True cluster count: **10** (9 attack types + normal).

---

## Root Cause

**HDBSCAN over-segmentation on balanced data.**

1. TON_IoT has 10 attack classes, 8 of which contain exactly 20,000 rows each. In a 10K eval subset, each class contributes ~950–1,000 rows. This is a highly balanced, large-class structure.

2. V3's `hdbscan_min_cluster_size=5` (from NSL-KDD config) is extremely permissive at this scale. HDBSCAN is splitting each 1,000-row true class into ~5–6 sub-clusters → **55 predicted clusters for 10 true classes**.

3. ARI penalises over-segmentation severely (ARI = 0.42 vs AMI = 0.71 — the gap is the over-segmentation signature). V3's HGNN embeddings contain strong discriminative signal (AMI=0.705 is competitive with K-Means AMI=0.769); the failure is in the clustering step, not the representation.

4. K-Means methods receive n_clusters=10 as input — exactly the true class count. Their advantage comes partly from knowing the number of clusters a priori. In a real deployment scenario (where the number of campaigns is unknown), K-Means would require a separate cluster-count selection step, which is not evaluated here.

---

## Graph Audit

- Missing `hostname`, `username` → no host/user entity nodes in the graph. The HGNN graph is IP-only (temporal + IP-shared-entity edges). This is a structural reduction compared to NSL-KDD/UNSW.
- `src_ip`, `dst_ip` present — IP-based entity edges still constructed.
- Impact is hard to isolate without an ablation, but the AMI=0.705 suggests the backbone still produces useful representations despite the sparser graph.

---

## Feature Audit

- Missing `src_bytes`, `dst_bytes`, `service` vs NSL-KDD/UNSW. Base features are fewer.
- `src_port`, `dst_port` are present (extra columns) but may not be used by the default MITRE-format feature extractor (which looks for `src_bytes`, `dst_bytes`).
- This might reduce the discriminative power of the base features, partially explaining why the HGNN embedding (which relies partly on raw features) is only marginally better than raw K-Means.

---

## Sweep Decision

**Rationale for running a targeted sweep:** The over-segmentation cause is clear and addressable. Increasing `hdbscan_min_cluster_size` through the full engine should reduce n_pred_clusters toward 10. This is not cherry-picking — we are correcting a parameter that was tuned for a different class-size regime.

**Sweep scope:** Vary `hdbscan_min_cluster_size` only (other parameters fixed). Must route through `benchmark/clustering_sweep_full_engine.py` (Phase 3 NSL-KDD lesson — standalone sweeps don't transfer).

**Sweep grid:**
- `hdbscan_min_cluster_size`: [5, 25, 50, 100, 200, 300]
- Other params: fixed at NSL-KDD defaults (pca=16, epsilon=0.1)
- Eval on dev split (seed 42) only; winner applied to eval split for final numbers.

**Honest cap:** If the swept V3 still loses to K-Means (raw) by >0.05 ARI, freeze with Path B result and document as a genuine finding.

---

## Sweep Results (full engine, dev split seed 42)

| mcs | ARI | AMI | n_clusters |
|-----|-----|-----|------------|
| 5 (default) | 0.380 | 0.714 | 63 |
| 25 | 0.381 | 0.716 | 44 |
| 50 | 0.379 | 0.713 | 38 |
| 100 | 0.383 | 0.716 | 33 |
| 200 | 0.480 | 0.744 | 21 |
| **300** | **0.531** | **0.751** | **17** |
| 400 | 0.507 | 0.736 | 13 |
| 500–1000 | 0.459 | 0.685 | 10 |

**Winner:** mcs=300 (dev ARI=0.531, 17 clusters).

## Sweep Validation on Eval Split (seed 142, seeds 42/43/44)

| Config | ARI | AMI | n_clusters |
|--------|-----|-----|------------|
| Default mcs=5 | 0.423 | 0.705 | 55 |
| Swept mcs=300 | 0.474 | 0.702 | 17 |
| K-Means (raw), n_clusters=10 | 0.622 | 0.769 | 10 |

**Conclusion:** Swept V3 (mcs=300) improves ARI by +0.051 but gap to K-Means (raw) remains 0.148 ARI. The honest cap (>0.05 gap) is exceeded. **Freeze with Path B result** — V3 loses on TON_IoT at v1.0.

## Important Nuance

K-Means receives `n_clusters=10` as input — the exact ground-truth class count. This is a form of privileged label information (knowing the number of clusters a priori). In a real deployment, this would require a separate cluster-count selection step. V3 with HDBSCAN is truly unsupervised with respect to cluster count; its 17-cluster prediction is an honest estimate, not a count-matched answer.
