# SQTK_SIEM — Label Audit (Stage 1)

**Date:** 2026-05-21
**Input:** `datasets/SQTK_SIEM/mitre_core_format.csv`

---

## File Provenance

| Field | Value |
|-------|-------|
| Row count | 5,100 |
| SHA256 | `ab2a63166dd79d570147de940edef242bb9e0304eccb1f79953c174874b5c617` |
| Format | CSV |
| Columns | 21 |

---

## Schema Comparison vs NSL-KDD / UNSW-NB15

| Column | NSL-KDD | UNSW | SQTK_SIEM | Notes |
|--------|---------|------|-----------|-------|
| timestamp | ✓ | ✓ | ✓ | |
| src_ip | ✓ | ✓ | ✓ | Used for IP-based graph edges |
| dst_ip | ✓ | ✓ | ✓ | |
| hostname | ✓ | ✓ | ✓ | Present (unlike TON_IoT) |
| username | ✓ | ✓ | ✓ | Present (unlike TON_IoT) |
| alert_type | ✓ | ✓ | ✓ | Native SIEM alert type — primary label |
| tactic | ✓ | ✓ | ✓ | 88.7% UNKNOWN — secondary track |
| campaign_id | ✓ | ✓ | ✓ | 89.4% UNKNOWN — secondary track |
| service | ✓ | ✓ | ✓ | |
| protocol | ✓ | ✓ | ✓ | |
| src_bytes | ✓ | ✓ | ✓ | |
| dst_bytes | ✓ | ✓ | ✓ | |
| stage | ✓ | ✓ | ✓ | |
| data_source | ✓ | ✓ | ✓ | |
| src_port | — | — | ✓ | Extra network feature |
| dst_port | — | — | ✓ | Extra network feature |
| device_type | — | — | ✓ | Extra feature |
| severity | — | — | ✓ | Extra feature |
| technique | — | — | ✓ | Extra feature |
| kcluster | — | — | ✓ | Pre-computed (11 clusters); reference only |

**Graph construction impact:** Full heterogeneous graph possible — hostname, username, src_ip, dst_ip all present. This is the dataset where heterogeneous graph structure is most likely to add real value.

---

## Label Track Summary

### `alert_type` — 14 classes (0 nulls) — **Recommended primary track**

| Value | Count | Notes |
|-------|-------|-------|
| Built connection | 1,252 | |
| Teardown connection | 1,217 | |
| passed | 827 | |
| Received | 509 | |
| blocked | 372 | |
| IDS:Reset | 184 | |
| Deny | 142 | |
| alerted | 129 | |
| alert | 128 | |
| NIL | 112 | |
| Reset | 97 | |
| denied | 81 | |
| notified | 42 | |
| permitted | 8 | Smallest class |

**Usability:** Native SIEM alert type. No nulls. 14 classes with wide imbalance (smallest = 8 rows). All 14 classes survive the full 5,100-row corpus.

### `tactic` — 9 classes (0 nulls, but UNKNOWN dominates)

| Value | Count | Notes |
|-------|-------|-------|
| UNKNOWN | 4,526 | 88.7% of corpus |
| RECONNAISSANCE | 293 | |
| INITIAL ACCESS | 88 | |
| EXFILTRATION | 58 | |
| DISCOVERY | 55 | |
| DEFENSE EVASION,PRIVILEGE ESCALATION | 31 | |
| RESOURCE DEVELOPMENT | 30 | |
| COMMAND AND CONTROL | 12 | |
| IMPACT | 7 | |

**Usability:** Valid secondary track despite UNKNOWN dominance. ARI scores will be low (UNKNOWN dominates), but included for completeness per lifecycle protocol.

### `campaign_id` — 8 classes (0 nulls, UNKNOWN dominates)

| Value | Count | Notes |
|-------|-------|-------|
| UNKNOWN | 4,557 | 89.4% of corpus |
| RECONNAISSANCE | 293 | |
| INITIAL ACCESS | 88 | |
| EXFILTRATION | 58 | |
| DISCOVERY | 55 | |
| RESOURCE DEVELOPMENT | 30 | |
| COMMAND AND CONTROL | 12 | |
| IMPACT | 7 | |

**Usability:** Secondary track. Same UNKNOWN dominance as tactic. campaign_id excludes the combined "DEFENSE EVASION,PRIVILEGE ESCALATION" row that tactic has (31 rows).

### `kcluster` — 11 clusters (pre-computed) — **EXCLUDED**

Pre-computed k-means clusters from prior analysis. Using this as an evaluation track would be circular (comparing V3 against another clustering algorithm's output, not ground truth). Documented as reference only.

---

## Small-Class Warnings

| Label track | Class | Count | % of corpus |
|------------|-------|-------|-------------|
| `alert_type` | permitted | 8 | 0.16% |
| `alert_type` | notified | 42 | 0.82% |
| `alert_type` | denied | 81 | 1.59% |
| `tactic` | IMPACT | 7 | 0.14% |
| `campaign_id` | IMPACT | 7 | 0.14% |

These are known dataset limitations, not benchmark errors.

---

## Graph Construction Verification

- src_ip: PRESENT ✓
- dst_ip: PRESENT ✓
- hostname: PRESENT ✓
- username: PRESENT ✓

Full heterogeneous graph construction possible — this is the most graph-rich dataset in the benchmark.

---

## Selected Label Tracks for Benchmark

| Track | Column | Classes | Notes |
|-------|--------|---------|-------|
| **Primary** | `alert_type` | 14 | Native SIEM alert type; 0 nulls |
| **Secondary** | `tactic` | 9 | UNKNOWN dominates (88.7%); low ARI expected |
| **Tertiary** | `campaign_id` | 8 | UNKNOWN dominates (89.4%); low ARI expected |

Excluded: `kcluster` (circular; prior clustering output).

---

## Stage 1 Exit Criterion

✅ At least one usable multi-class label track confirmed (`alert_type`, 14 classes).
✅ Label audit written with row count, SHA256, all candidate tracks documented.
✅ Schema differences from NSL-KDD/UNSW documented.
✅ Decision: 3 benchmark tracks selected, 1 excluded with rationale.
