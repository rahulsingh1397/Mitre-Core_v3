# NSL-KDD — Decision Log

Chronological record of non-trivial choices made during the NSL-KDD lifecycle.
Each entry: date, choice made, alternatives considered, rationale.

---

## 2026-05-17 — Choose 10K stratified sample as benchmark size
**Choice:** 10,000-row stratified sample from 125,973 total rows.
**Alternatives considered:** full dataset (125,973 rows); 5,000-row sample.
**Rationale:** 10K balances statistical reliability, runtime on CPU, and reproducibility; stratified sampling preserves class balance; matches existing V2 sanity-check convention.

## 2026-05-17 — Fix unfrozen sampling as first protocol improvement
**Choice:** add `sample_seed: 42` to freeze the sampled evaluation subset.
**Alternatives:** run on the full dataset (no sampling); use a different seed.
**Rationale:** unfrozen sampling caused inflated cross-seed variance that mixed data-drift with method variance; freezing isolates the correct source of variance. Seed 42 is the established repo convention.

## 2026-05-17 — Separate dev (seed 42) from eval (seed 142)
**Choice:** dev subset (seed 42) for tuning; eval subset (seed 142, disjoint) for reported numbers.
**Alternatives:** single frozen subset for both.
**Rationale:** single-subset tuning creates implicit overfitting to evaluation data; separation follows Bilot-et-al fair-evaluation protocol.

## 2026-05-17 — Revert Phase 3 sweep winner; lock original engine config
**Choice:** discard Phase 3 HDBSCAN sweep winner (min_cluster_size=10, pca=8, epsilon=0.0); use original config (mcs=5, pca=16, eps=0.1).
**Alternatives:** accept sweep winner.
**Rationale:** standalone sweep produced catastrophic ARI degradation (0.632 → 0.078) when ported to the full V3 engine. Root cause: standalone HDBSCAN does not replicate EmbeddingConfidenceScorer pipeline. Original config was validated end-to-end. See `docs/lessons/phase3_sweep_methodology.md`.

## 2026-05-17 — Demote `attack_f1`; add `attack_f1_demoted`
**Choice:** add `attack_f1_demoted` metric that zeros attack_f1 when ≤1 non-noise cluster.
**Alternatives:** remove attack_f1 entirely.
**Rationale:** attack_f1 = 1.0 for all methods on NSL-KDD; it is non-discriminative. `attack_f1_demoted` retains the metric while correctly flagging trivial clusterings (e.g., DBSCAN raw with 100% noise).

## 2026-05-17 — Use `tactic` as primary headline label track
**Choice:** headline results on `tactic` (8 classes); record `alert_type` and `campaign_id` as indicative dev-only tracks.
**Alternatives:** use `campaign_id` as primary (shows higher ARI); use `alert_type` for binary comparison.
**Rationale:** `tactic` is the MITRE ATT&CK semantic alignment most relevant to SOC use cases. `campaign_id` may be a cleaner clustering target but is less semantically meaningful as a threat-correlation label. Noting the `campaign_id` advantage as a finding for future v1.1.

## 2026-05-19 — Freeze v1.0
**Choice:** lock Phase 5 result as the immutable v1.0 baseline.
**Alternatives:** continue tuning before freezing.
**Rationale:** V3 holds a +0.19 ARI margin over the best baseline (Spectral raw). Per master plan decision gate: margin > 0.1 → freeze immediately. Further tuning without running UNSW-NB15 would narrow the scope of the contribution; the multi-dataset story is more valuable.
