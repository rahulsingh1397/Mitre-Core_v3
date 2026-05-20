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

---

## 2026-05-20 — Path B: V3 loses on TON_IoT (Stage 4 — Decision Gate)

**Choice:** Trigger Path B investigation rather than freezing immediately.

**Trigger:** V3 ARI=0.423 vs K-Means(raw) ARI=0.622 on `alert_type` track — margin=-0.199.

**Alternatives considered:**
- Accept the result immediately (no investigation). Rejected: the margin is large; a root-cause diagnosis was required to confirm the loss is genuine (not a misconfiguration).

**Rationale:** The Path B investigation revealed HDBSCAN over-segmentation as the root cause (55 predicted clusters for 10 true classes, mcs=5 too permissive for 1K-per-class balanced data). This is a legitimate addressable cause — a targeted sweep was warranted.

**Pointer:** `docs/datasets/ton_iot/investigation.md`

---

## 2026-05-20 — Full-engine sweep: mcs grid [5, 25, 50, 100, 200, 300, 400, 500–1000]

**Choice:** Sweep `hdbscan_min_cluster_size` only, all other params fixed at NSL-KDD defaults.

**Alternatives considered:**
- Sweep multiple parameters simultaneously (pca_components, epsilon). Rejected: increases search space unnecessarily; the investigation identified mcs as the single causal parameter.
- Use standalone `clustering_sweep.py`. Rejected: Phase 3 NSL-KDD lesson — standalone sweeps don't transfer to the full V3 pipeline. Used `clustering_sweep_full_engine.py` instead.

**Rationale:** The grid was intentionally wide (5→1000) to map the full mcs landscape rather than stopping at the first improvement. Winner: mcs=300 (dev ARI=0.531, 17 clusters). Applied to eval: ARI=0.474 (gap to K-Means=0.148 > honest cap 0.05).

**Pointer:** `docs/datasets/ton_iot/investigation.md` — sweep table

---

## 2026-05-20 — Freeze with Path B result despite sweep improvement

**Choice:** Freeze the **default mcs=5 config** result (ARI=0.423) as the v1.0 baseline, not the swept winner.

**Alternatives considered:**
- Freeze with swept mcs=300 result (ARI=0.474). Rejected: the default config is used for all datasets for cross-dataset consistency; sweeping one dataset mid-lifecycle would break the fair-comparison protocol.

**Rationale:** The swept mcs=300 result is documented in `investigation.md` as supplemental. The honest cap (gap > 0.05 ARI even after the best sweep) confirms V3 genuinely loses on TON_IoT at v1.0. This is a credible, publishable negative result.

**Pointer:** `docs/datasets/ton_iot/v1.0_baseline.md`

---

## 2026-05-20 — dominant_confusion_accuracy demoted (Stage 5)

**Choice:** Demote `dominant_confusion_accuracy` from the active metric set for TON_IoT.

**Rationale:** Constant 1.0 across all 10 methods — third consecutive dataset (after NSL-KDD and UNSW-NB15) showing this behavior. The metric is structurally degenerate on well-separated multi-class data, not a dataset-specific anomaly.

**Pointer:** `learnings.md` L5 — recommended global pre-demotion for future datasets.

---

## 2026-05-20 — v1.0 Freeze (Stage 6)

**Choice:** Freeze all artifacts to `benchmark/results/frozen/ton_iot/v1.0/`.

**Outcome:** V3 ARI=0.423±0.000 (alert_type, seeds 42/43/44). K-Means(raw) wins at ARI=0.622±0.033. Git tag: `ton-iot-v1.0`.

**Pointer:** `docs/datasets/ton_iot/v1.0_baseline.md`
