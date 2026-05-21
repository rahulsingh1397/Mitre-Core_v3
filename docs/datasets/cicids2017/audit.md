# CICIDS2017 — Label Audit

**Script:** `scripts/audit_cicids2017_labels.py`
**File:** `datasets/CICIDS2017/mitre_format.parquet`
**SHA256:** `ee8c20cb02ee46f5471b4473e28247a4abb63aebea3e839264a15f2bebd604c2`
**Total rows:** 3,119,345
**Columns (10):** `timestamp`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `tactic`, `campaign_id`, `alert_type`, `label`

---

## Schema Diff vs NSL-KDD/UNSW Standard

| Status | Column |
|---|---|
| Missing | `hostname`, `username`, `attack_cat`, `service`, `src_bytes`, `dst_bytes`, `stage`, `data_source` |
| Extra | `src_port`, `dst_port`, `label` (binary int 0/1) |

Note: IP-only graph — same profile as TON_IoT. `AlertToGraphConverter` handles gracefully; only
temporal edges are built (no `shares_ip` edges — those require `SourceAddress`/`DestinationAddress`
column names which CICIDS2017 does not provide).

---

## Data Quality Issue — Null alert_type Rows

- **Count:** 288,602 rows (9.25% of dataset)
- **Pattern:** all four fields are null simultaneously — `timestamp`, `src_ip`, `dst_ip`, `alert_type`
- **Marker:** `campaign_id = 99`, `label = 1` (attack events with no assigned category)
- **Root cause:** CICIDS2017 source CSVs contain attack flows captured but not categorized by the
  traffic generator; this is a known data quality issue in the original dataset.

**Benchmark treatment:** The benchmark's `stratified_sample: true` path fills null labels with
`"UNKNOWN"` before stratified sampling (`df[label_col].fillna("UNKNOWN")`). These rows are NOT
excluded — they are sampled proportionally and appear as an "UNKNOWN" class in the 10K subset
(~925 rows). In the graph, they become isolated alert nodes (no IP edges due to null src_ip/dst_ip).
This matches the behavior observed on prior datasets and is correct.

**Impact on evaluation:** "UNKNOWN" rows contribute a 13th class to alert_type evaluation.
Metrics will be slightly penalized because cluster assignments for isolated nodes are arbitrary.
This is documented rather than worked around — the penalty is real and reflects the dataset's
inherent data quality.

---

## Label Track: alert_type (15 classes, 288,602 nulls)

| Class | Count |
|---|---|
| BENIGN | 2,273,097 |
| DoS Hulk | 231,073 |
| PortScan | 158,930 |
| DDoS | 128,027 |
| DoS GoldenEye | 10,293 |
| FTP-Patator | 7,938 |
| SSH-Patator | 5,897 |
| DoS slowloris | 5,796 |
| DoS Slowhttptest | 5,499 |
| Bot | 1,966 |
| Web Attack - Brute Force | 1,507 |
| Web Attack - XSS | 652 |
| Infiltration | 36 |
| Web Attack - Sql Injection | 21 |
| Heartbleed | 11 |
| NULL (UNKNOWN) | 288,602 |

**Total named + null:** 3,119,345 ✓

**Imbalance:** BENIGN = 72.9% of all rows; 84.5% of named rows.

---

## Label Track: tactic (7 classes, 2,561,699 nulls)

Nulls = BENIGN rows (no tactic label) + 288,602 untyped attack rows.

| Class | Count |
|---|---|
| Impact | 381,610 |
| Reconnaissance | 158,930 |
| CredentialAccess | 15,835 |
| CommandAndControl | 1,966 |
| Execution | 673 |
| LateralMovement | 36 |
| InitialAccess | 11 |
| NULL | 2,561,699 |

Note: `tactic` nulls cannot be filled with "UNKNOWN" cleanly for benchmark use because the
majority of rows are legitimately non-tactics (BENIGN). Using `tactic` as `label_col` would
produce a heavily "UNKNOWN"-dominated sample. It is retained as an `alt_label_col` only.

---

## Label Track: campaign_id (16 IDs + sentinel)

| ID | Meaning | Count |
|---|---|---|
| 0 | BENIGN | 2,273,097 |
| 99 | Untyped attack (no alert_type) | 288,602 |
| 1–15 | Named attack categories | 557,646 total |

---

## Small Class Warnings

**Corpus-level (<20 rows):**
- `Heartbleed`: 11 rows
- `Web Attack - Sql Injection`: 21 rows

**Near-zero in 10K subset (<1.0 expected row):**

| Class | Corpus count | Expected in 10K |
|---|---|---|
| Heartbleed | 11 | 0.04 |
| Web Attack - Sql Injection | 21 | 0.07 |
| Infiltration | 36 | 0.12 |

These three classes are **absent from every 10K stratified sample**. This is a known limitation
of the dataset, not a benchmark error. They are excluded from per-class analysis but do not
prevent multi-class evaluation (12 evaluable named classes remain).

---

## 10K Stratified Subset (seed 42, stratified by alert_type incl. UNKNOWN)

**13 classes present** in the 10K subset (12 named + 1 UNKNOWN):

| Class | Approx. count |
|---|---|
| BENIGN | ~7,289 |
| DoS Hulk | ~741 |
| PortScan | ~510 |
| DDoS | ~410 |
| UNKNOWN (null alert_type) | ~925 |
| DoS GoldenEye | ~33 |
| FTP-Patator | ~25 |
| SSH-Patator | ~19 |
| DoS Slowhttptest | ~18 |
| DoS slowloris | ~19 |
| Bot | ~6 |
| Web Attack - Brute Force | ~5 |
| Web Attack - XSS | ~2 |

3 classes absent (Infiltration, SQL Injection, Heartbleed — expected < 1 row each).
2 tactic classes absent in tactic track (LateralMovement=36 → ~1.2 expected; InitialAccess=11 → ~0.04).

---

## Graph Construction Feasibility

| Column | Status | Notes |
|---|---|---|
| `src_ip` | PRESENT | Used for temporal edge grouping and node features |
| `dst_ip` | PRESENT | Used for node features |
| `hostname` | absent | Not in dataset |
| `username` | absent | Not in dataset |
| `SourceAddress` | absent | Required for `shares_ip` edges — not present; edge type skipped |
| `DestinationAddress` | absent | Same |

**Graph structure for CICIDS2017:** Alert nodes connected by temporal edges only (same as TON_IoT).
No heterogeneous IP/host/user nodes are built. The GNN still processes alert-level features
(protocol, tactic frequency, IP frequency counts, temporal density) which capture attack semantics.

---

## Selected Label Tracks for Benchmark

| Role | Column | Classes (10K) | Notes |
|---|---|---|---|
| Primary | `alert_type` | 13 (12 named + UNKNOWN) | 3 corpus-sparse classes absent |
| Secondary | `tactic` | 5 | BENIGN + untyped rows have null tactic |
| Tertiary | `campaign_id` | ~15 | Numeric IDs; 0=BENIGN, 99=untyped |

---

## Usability Verdict

**USABLE.** `alert_type` provides 12 evaluable named classes in the 10K subset, sufficient for
multi-class clustering evaluation. The "UNKNOWN" class (~925 rows) adds a 13th class from null
rows. Sparse-class limitation (3 classes absent) and null-row treatment are documented above.

The dataset is the fourth in the benchmark and confirms network-IDS claim across a modern,
large-scale dataset with a different traffic profile than NSL-KDD/UNSW-NB15.
