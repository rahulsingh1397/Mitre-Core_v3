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

---

## 2026-05-19 — Stage 4: Path A (skip sweep)

**Choice:** Proceed directly to Stage 5 without clustering hyperparameter sweep.

**Alternatives considered:**
- Path B: investigate + sweep via `benchmark/clustering_sweep_full_engine.py`

**Rationale:** V3 ARI on attack_cat = 0.5639 vs best baseline (PCA+HDBSCAN) = 0.3543. Margin = +0.2096, well above the >0.1 threshold for Path A. Sweep is not warranted.

**Pointer:** `benchmark/results/latest/unsw_nb15/baseline_roster_summary.csv`

---

## 2026-05-19 — Stage 5: dominant_confusion_accuracy demoted

**Choice:** Demote `dominant_confusion_accuracy` from UNSW-NB15 reporting (constant 1.0 across all 10 methods).

**Alternatives considered:** Keep it and note it's uninformative.

**Rationale:** A metric constant across all methods is not discriminative and misleads readers about relative performance. Same demotion logic as `attack_f1` on NSL-KDD.

**Pointer:** `benchmark/results/latest/unsw_nb15/baseline_roster_summary.csv` — dominant_confusion_accuracy column, all values 1.0.

---

## 2026-05-19 — Freeze v1.0

**Choice:** Freeze UNSW-NB15 results as v1.0 at this point.

**Rationale:** All 6 lifecycle stages complete. V3 maintains a strong margin (+0.21 ARI on attack_cat). No additional tuning or sweep was needed.

**Pointer:** `benchmark/results/frozen/unsw_nb15/v1.0/`
