# Exp 2.5 — GMM+BIC Clustering Swap (Finding Doc)

**Date:** 2026-05-23
**Status:** PARTIAL SUCCESS — TON-IoT freeze v1.1; CICIDS2017 fail, retry queued

---

## Hypothesis Tested (H1)

V3's HDBSCAN cannot discover cluster count from density on datasets where embeddings are diverse (CICIDS2017) or where true cluster structure is not density-shaped (TON-IoT). Replacing HDBSCAN with GMM+BIC, which selects k by minimizing Bayesian Information Criterion, lets the clusterer use the GNN's full embedding signal without needing oracle k.

---

## Implementation

- `EmbeddingConfidenceScorer.__init__`: added `gmm_bic_k_min`, `gmm_bic_k_max` params
- `EmbeddingConfidenceScorer.fit_score`: added `elif clustering_method == "gmm"` branch with BIC sweep
- `HGNNCorrelationEngine.__init__`: passes the two new params through
- `benchmark/datasets_real.yaml`: added `CICIDS2017-exp2.5` and `TON-IoT-exp2.5` entries

---

## Results

| Dataset | v1.0 ARI | Exp 2.5 ARI | Δ | BIC-selected k | True k | Decision |
|---|---|---|---|---|---|---|
| TON-IoT | 0.423 | **0.604** | **+0.181** | 20 | 10 | ✅ FREEZE v1.1 |
| CICIDS2017 | 0.177 | 0.082 | −0.095 | 30 (cap) | 15 | ❌ FAIL — retry with lower k_max |

---

## Findings

1. **H1 confirmed on TON-IoT.** V3 was leaving 0.18 ARI on the table because of HDBSCAN's k-discovery. GMM+BIC closes the v1.0 gap from −0.199 to −0.018 vs K-Means(raw).
2. **CICIDS2017 BIC saturated at k_max=30.** Full-covariance GMM over-segments on imbalanced data when given enough room. Retry with `gmm_bic_k_max: 15` (matches true class count) queued as Exp 2.5b.
3. **Pure-inference fix.** No retraining required — same checkpoint, same splits, same engine. Only `engine_kwargs.clustering_method` changes.

---

## Artifacts

- TON-IoT: `benchmark/results/frozen/ton_iot/v1.1/` 
- CICIDS2017 (failed): `benchmark/results/latest/exp2.5_gmm/results.csv` 
- Code: `hgnn/hgnn_correlation.py` GMM branch

---

## Update 2026-05-23 — Exp 2.5b CICIDS2017 Retry

Retried CICIDS2017 with `gmm_bic_k_max: 15` (matches true class count) to address the v2.5 BIC saturation at k=30.

| Run | k_max | BIC-selected k | V3 ARI |
|---|---|---|---|
| Exp 2.5 | 30 | 30 (saturated) | 0.082 |
| Exp 2.5b | 15 | 14 (deterministic) | 0.1737 |
| v1.0 (HDBSCAN) | n/a | n/a | 0.1771 |

**Decision: REJECTED.** With k_max=15, BIC selected k=14 cleanly (one below true k) — no saturation. Yet ARI is essentially unchanged from v1.0 HDBSCAN (Δ=−0.0033). This is the cleanest possible refutation of the clusterer-bottleneck hypothesis for CICIDS2017: when both HDBSCAN and a deterministic-k=14 GMM produce the same ARI, the clusterer is exonerated. The 72.9% BENIGN class lacks discoverable sub-structure in the embedding space. Move to embedding-targeted experiments (Exp 3 / Exp 4).
