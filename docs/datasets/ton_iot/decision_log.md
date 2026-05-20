# TON_IoT — Decision Log

Format: `## YYYY-MM-DD — <Choice made>`

---

## 2026-05-20 — Primary label track: alert_type

**Choice:** Use `alert_type` as the primary benchmark label track.

**Alternatives considered:**
- `tactic` (7 classes after NaN-fill): coarser — multiple attack types map to same tactic (ddos+dos+ransomware → Impact).
- `campaign_id` (10 classes): excluded — 1-to-1 duplicate of alert_type.

**Rationale:** `alert_type` is the native TON_IoT label and most fine-grained (10 classes, 0 nulls). Directly names attack types in published TON_IoT papers.

**Pointer:** `docs/datasets/ton_iot/audit.md` — cross-tab confirms alert_type vs tactic mapping.

---

## 2026-05-20 — Binary track: use `alert_type` "normal" class (not `label` column)

**Choice:** Use `alert_type` as the label column for the benchmark (including binary comparison). The numeric `label` column is available but `alert_type="normal"` directly encodes the benign class.

**Alternatives considered:**
- Use `label` (int 0/1) as a separate binary track via `alt_label_cols`.

**Rationale:** Keeping `alert_type` as the single label column is simpler and consistent with NSL-KDD/UNSW. The benchmark already handles `benign_label="normal"` to extract binary attack/normal signal from the primary multi-class label. The numeric `label` column would require special handling in the metric computation.

**Pointer:** `docs/datasets/ton_iot/protocol.md`

---

## 2026-05-20 — No parquet → CSV conversion needed

**Choice:** Use `datasets/TON_IoT/mitre_format.parquet` directly (no conversion to CSV).

**Alternatives considered:**
- Convert parquet to CSV for consistency with NSL-KDD/UNSW.

**Rationale:** `benchmark.py` already handles parquet (line 259-262). V3 smoke test confirmed the full pipeline works on the parquet schema. No conversion needed.

**Pointer:** `docs/datasets/ton_iot/audit.md` — graph construction feasibility section.

---

## 2026-05-20 — Sample size: 10,000 (default)

**Choice:** 10,000-row stratified sample, matching NSL-KDD/UNSW default.

**Rationale:** Consistent cross-dataset comparison. TON_IoT has 211,043 rows; 10K is proportionally reasonable. The mitm class (1,043 rows, ~49 in 10K) is evaluable (unlike Worms at ~7 in UNSW). No need to increase sample size.

**Pointer:** Master plan Part IV.3 — "Expected effort: 2–3 days (loader is the heavy part)"; loader was simpler than expected.
