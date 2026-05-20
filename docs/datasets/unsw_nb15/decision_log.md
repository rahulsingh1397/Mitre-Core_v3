# UNSW-NB15 — Decision Log

Format: `## YYYY-MM-DD — <Choice made>`
Include: alternatives considered, rationale, pointer to artifact that motivated the choice.

---

## 2026-05-19 — Primary label track: attack_cat over tactic

**Choice:** Use `attack_cat` as the primary benchmark label track (instead of `tactic`).

**Alternatives considered:**
- `tactic` (7+1 classes after NaN-fill) — mirrors NSL-KDD primary track for direct comparison.
- `attack_cat` (10 classes, 0 nulls) — native UNSW-NB15 label; more fine-grained; directly cited in the UNSW-NB15 paper.

**Rationale:** `attack_cat` is the authoritative UNSW label and the column cited in published UNSW-NB15 papers. It is more fine-grained (10 classes vs 8), has no null values, and is directly interpretable. `tactic` is included as secondary to allow cross-dataset comparison with NSL-KDD.

**Pointer:** `docs/datasets/unsw_nb15/audit.md` — attack_cat vs tactic cross-tab confirms 1-to-1 alignment.

---

## 2026-05-19 — Excluded campaign_id and stage from benchmark

**Choice:** Exclude `campaign_id` and `stage` from benchmark label tracks.

**Alternatives considered:**
- Include `campaign_id` as a third multi-class track (9 classes).
- Include `stage` as a temporal-phase track (7 classes).

**Rationale:**
- `campaign_id` is essentially a coarsening of `attack_cat` (Normal+Worms merged into one campaign_id). Evaluating on both would produce near-identical metrics with no additional signal.
- `stage` represents temporal attack phases (Stage_0 through Stage_4), not attack types. The phases are roughly uniformly distributed across attack traffic, making them poor targets for attack-type clustering. Unlike NSL-KDD where `stage` was redundant with `alert_type`, here it is distinct but semantically uninformative for the clustering objective.

**Pointer:** `docs/datasets/unsw_nb15/audit.md` — stage cross-tab shows ~21,700 rows per stage level.

---

## 2026-05-19 — Sample size: 10,000 (default, matching NSL-KDD)

**Choice:** Use 10,000-row stratified sample, matching NSL-KDD default.

**Alternatives considered:**
- 20,000 rows (allowed by master plan if dataset is large) — UNSW has 175,341 rows.

**Rationale:** Keeping 10,000 makes cross-dataset ARI comparisons directly interpretable and halves compute cost. The Worms class caveat (130 rows full corpus, ~7 in 10K subset) is documented as a known limitation rather than a reason to increase sample size.

**Pointer:** Master plan Part VII.2, Stage 2 note on sample size.
