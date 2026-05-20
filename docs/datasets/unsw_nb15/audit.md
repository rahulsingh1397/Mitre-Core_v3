# UNSW-NB15 — Label Audit (Stage 1)

**Date:** 2026-05-19
**Script:** `scripts/audit_unsw_nb15_labels.py`
**Input:** `datasets/unsw_nb15/mitre_format.csv`

---

## File Provenance

| Field | Value |
|-------|-------|
| Row count | 175,341 |
| SHA256 | `c7856d8428fd7b35ffd233ccece378be3e0b2ba9d23c6b7bfe37dab13441b892` |
| Columns | 15 (same MITRE-format schema as NSL-KDD) |
| Schema | timestamp, src_ip, dst_ip, hostname, username, alert_type, tactic, campaign_id, attack_cat, protocol, service, src_bytes, dst_bytes, stage, data_source |

---

## Label Track Summary

### `tactic` — 7 attack classes + NaN for Normal rows

| Value | Count | Notes |
|-------|-------|-------|
| NaN (Normal) | 56,000 | All normal rows; must fill NaN → "Normal" before use |
| Initial Access | 73,393 | Largest attack class |
| Execution | 19,317 | |
| Impact | 12,264 | |
| Reconnaissance | 10,491 | |
| Collection | 2,000 | |
| Persistence | 1,746 | |
| Lateral Movement | 130 | Smallest class |

**Usability:** Valid multi-class track (8 effective classes after NaN-fill). Same NaN pattern as NSL-KDD. The `benign_label` YAML config and the benchmark's NaN-fill logic handle this correctly — no code change needed.

### `alert_type` — Binary (0 nulls)

| Value | Count |
|-------|-------|
| attack | 119,341 (68.1%) |
| normal | 56,000 (31.9%) |

**Usability:** Clean binary track. Best suited for binary_ari metric.

### `attack_cat` — 10 classes (0 nulls) — **Recommended primary track**

| Value | Count |
|-------|-------|
| Normal | 56,000 |
| Generic | 40,000 |
| Exploits | 33,393 |
| Fuzzers | 18,184 |
| DoS | 12,264 |
| Reconnaissance | 10,491 |
| Analysis | 2,000 |
| Backdoor | 1,746 |
| Shellcode | 1,133 |
| Worms | 130 |

**Usability:** Most informative fine-grained track. No nulls. Directly names UNSW attack categories (the native label column from the original UNSW-NB15 paper). All 10 classes survive a 10K sample (seed 42). **Warning:** Worms has only 130 rows in the full corpus; in a 10K subset expect ~7 rows — very small class; document this limitation.

### `campaign_id` — 9 classes (0 nulls)

Maps loosely to `attack_cat`: campaign_id 22 ≈ Normal (+130 Worms), 46 = Generic, 34 = Exploits, 33 = Fuzzers, 38 = DoS, 26 = Reconnaissance, 3 = Analysis, 17 = Backdoor, 14 = Shellcode. One-to-one with attack_cat minus the Normal/Worms merge.

**Usability:** Near-duplicate of `attack_cat`. **Excluded from benchmark** to avoid redundancy.

### `stage` — 7 classes (0 nulls)

| Value | Count | Notes |
|-------|-------|-------|
| Normal | 56,000 | Normal rows |
| Stage_2 | 21,837 | Temporal attack phases |
| Stage_0 | 21,815 | Nearly uniform across attack rows |
| Stage_3 | 21,760 | |
| Stage_1 | 21,724 | |
| Stage_4 | 21,714 | |
| Initial Discovery | 10,491 | Reconnaissance class |

**Usability:** Not an attack-category label — represents temporal phases of the kill chain. Unlike NSL-KDD where `stage` was redundant with `alert_type`, here it is distinct but *not semantically meaningful* for clustering evaluation (attack stages 0–4 are nearly uniform splits of attack traffic, not attack types). **Excluded from benchmark.**

---

## Cross-tabulation Findings

- `tactic` and `alert_type` are perfectly aligned: all 56,000 normal rows have NaN tactic; all 119,341 attack rows have a non-null tactic. No overlap or inconsistency.
- `attack_cat` and `tactic` are in 1-to-1 alignment (each attack_cat value maps to exactly one tactic value — confirmed by cross-tab).
- `campaign_id` is essentially `attack_cat` with the Normal and Worms rows merged into campaign_id 22.

---

## Small-Class Warning

| Label track | Class | Count | Expected in 10K subset |
|------------|-------|-------|------------------------|
| `attack_cat` / `tactic` | Worms / Lateral Movement | 130 | ~7 rows |
| `attack_cat` | Shellcode | 1,133 | ~65 rows |

Worms (130 rows, 0.074% of corpus) will have very sparse representation in a 10K subset. This class is likely not evaluable in isolation — it will mostly be absorbed into a larger cluster. Document as a known limitation in `v1.0_baseline.md`.

---

## 10K Subset Check (seed 42)

All four candidate label tracks (`tactic`, `alert_type`, `campaign_id`, `attack_cat`) preserve all their unique classes in a 10,000-row stratified sample. No class is entirely absent.

---

## Selected Label Tracks for Benchmark

| Track | Rationale |
|-------|-----------|
| `attack_cat` | **Primary.** 10 classes, 0 nulls, direct UNSW attack-type label. Most informative for clustering evaluation. |
| `tactic` | **Secondary.** 8 effective classes (NaN→"Normal" fill). Mirrors NSL-KDD primary for cross-dataset comparison. |
| `alert_type` | **Tertiary.** Binary track for binary_ari headline. |

Excluded: `campaign_id` (redundant with attack_cat), `stage` (temporal phases, not attack types).

---

## Stage 1 Exit Criterion

✅ At least one usable multi-class label track confirmed (`attack_cat`, `tactic`).
✅ Label audit written with row count, SHA256, all candidate tracks documented.
✅ Decision: 3 benchmark tracks selected, 2 excluded with rationale.
