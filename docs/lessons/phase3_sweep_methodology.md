# Lesson: Standalone Clustering Sweeps Don't Transfer to the Full Engine

**Date discovered:** 2026-05-17 (NSL-KDD Phase 3)
**Severity:** High — produced a winner that when applied caused catastrophic regression
**Status:** Documented, methodology fixed

---

## What happened

Phase 3 ran a HDBSCAN grid sweep on cached HGNN embeddings extracted from the V3 engine.
The sweep was conducted in standalone mode: embeddings extracted once, HDBSCAN run
directly on them with varying hyperparameters.

**Sweep winner:** `min_cluster_size=10, pca_components=8, epsilon=0.0`
**Dev ARI on standalone HDBSCAN:** 0.338

When this winner was applied via the full `V3CorrelationEngine`:

| Metric | Before (original config) | After (sweep winner) |
|---|---|---|
| tactic ARI | 0.632 | 0.078 |
| alert_type ARI | 0.501 | 0.026 |
| campaign_id ARI | 0.675 | 0.090 |

ARI dropped by 85–95%. The sweep winner was immediately reverted.

---

## Root cause

`V3CorrelationEngine` does not run HDBSCAN directly on raw HGNN embeddings.
The full pipeline applies:

1. HGNN inference → raw embeddings
2. `EmbeddingConfidenceScorer` post-processing (GAEC scoring, optional raw-feature
   concatenation, PCA, whitening/normalization)
3. HDBSCAN on the post-processed embedding space

The standalone sweep ran HDBSCAN on step-1 outputs, not step-3 inputs. The effective
geometry seen by HDBSCAN is fundamentally different in the two cases.
A config tuned for step-1 space cannot be expected to work in step-3 space.

---

## Fix

**Do not use `benchmark/clustering_sweep_standalone.py` for sweeps whose winners
will be applied to the production pipeline.**

Use `benchmark/clustering_sweep_full_engine.py` instead. It routes every config
through the complete `V3CorrelationEngine.cluster_alerts(...)` call, so the
clustering hyperparameters are optimized for the actual inference geometry.

`clustering_sweep_standalone.py` has been renamed and annotated with a warning banner.
It is retained only as historical reference of what not to do.

---

## Rule for future sweeps

> Every clustering sweep whose winner will be applied to V3 must run the
> full engine. Standalone sweeps on cached embeddings are only valid for
> understanding the embedding space, never for config selection.

This rule is recorded in `docs/datasets/nsl_kdd/learnings.md` (L2) and should be
checked at the start of any future sweep phase for any dataset.
