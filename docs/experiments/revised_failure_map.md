# Revised V3 Failure Map

**Date:** 2026-05-23
**Supersedes:** Part X.0 (original failure map, now invalidated by Exp 1 + Exp 2 S1 results)

---

## Evidence Base

| Experiment | Outcome | What it ruled out |
|---|---|---|
| Exp 1 — Hard negative mining | REJECTED | Loss-only intervention insufficient |
| Exp 2 S1 — Collapse diagnostic | REJECTED | No embedding collapse exists anywhere |

---

## Corrected Per-Dataset Analysis

| Dataset | V3 v1.0 ARI | Cosine_sim | Best baseline | Baseline advantage | Root cause |
|---|---|---|---|---|---|
| TON_IoT | 0.423 (−0.199) | 0.73 | K-Means(raw) 0.622 | Oracle n_clusters=10 | HDBSCAN cannot discover k density modes |
| CICIDS2017 | 0.177 (−0.156) | **0.60** | Spectral(emb) 0.333 | Oracle n_clusters=15 | Embeddings too diverse for HDBSCAN density-finding |
| SQTK_SIEM | 0.355 (−0.027) | 0.79 | PCA+HDBSCAN 0.382 | PCA before HDBSCAN | Preprocessing gap: PCA reduces to fewer dims before HDBSCAN |
| NSL-KDD | 0.602 (+0.188) | 0.70 | Spectral(raw) 0.420 | None — V3 wins | n/a |
| UNSW-NB15 | 0.564 (+0.210) | 0.74 | PCA+HDBSCAN 0.354 | None — V3 wins | n/a |
| DARPA OpTC | 0.999 binary | — | Tied | None | n/a — binary task at ceiling |

---

## Three Hypotheses

**H1 — Cluster-count discovery is the real bottleneck**
All loser-beating methods have oracle `n_clusters`. HDBSCAN discovers k from density. When
embeddings are diverse (CICIDS2017 cosine_sim=0.60), no neighborhood reaches mcs=5, and
HDBSCAN over-segments. Fix: replace HDBSCAN with a k-selection algorithm (GMM+BIC).

**H2 — IP-only graphs lose structural signal**
CICIDS2017 and TON_IoT have no hostname/username edges. Only `shares_ip` and `temporal_near`
are active (2 of 30 defined edge relation types). HGT type-specific attention cannot exploit
what isn't there — but structural positional encoding (Exp 5) could re-inject signal.

**H3 — Winners win because their graphs are fully heterogeneous**
NSL-KDD and UNSW-NB15 have full hostname + username + IP structure. Per-relation attention
is meaningful. "V3 works on classic IDS" really means "V3 works when the graph is actually
heterogeneous."

---

## Revised Experiment Lineup

| # | Experiment | GPU cost | Targets | Hypothesis |
|---|---|---|---|---|
| ✅ Exp 1 | Hard neg mining | 4h | CICIDS, TON | REJECTED |
| ✅ Exp 2 S1 | Collapse diagnostic | 0 | All 5 | REJECTED; found script bug |
| **🆕 Exp 2.5** | HDBSCAN → GMM+BIC | 0 | TON_IoT, CICIDS | H1 |
| **🆕 Exp 2.6** | V3 emb → PCA(20) → HDBSCAN | 0 | SQTK_SIEM | H1 (preprocessing) |
| Exp 3 | 15-dim features | ~4h | CICIDS, TON | Anchor embedding diversity |
| Exp 4 | HGT architecture | ~6h | NSL-KDD, UNSW | H3 (push winners) |
| Exp 5 | Heterogeneous PE | ~3h | CICIDS, TON | H2 (sparse-graph) |

**Decision gate:** if Exp 2.5 or 2.6 produces ≥+0.05 ARI on any loser → freeze v1.1 immediately
(pure engine_kwargs change, no retraining). If not, escalate to Exp 3.
