# TON_IoT — Label Audit (Stage 1)

**Date:** 2026-05-20
**Script:** `scripts/audit_ton_iot_labels.py`
**Input:** `datasets/TON_IoT/mitre_format.parquet`

---

## File Provenance

| Field | Value |
|-------|-------|
| Row count | 211,043 |
| SHA256 | `0d307cb86b64099efb13088d94096a7863f3b5500396887eab437fb88ca0ce6f` |
| Format | Parquet (benchmark.py already supports parquet at line 259) |
| Columns | 11 |

---

## Schema Comparison vs NSL-KDD / UNSW-NB15

| Column | NSL-KDD | UNSW | TON_IoT | Notes |
|--------|---------|------|---------|-------|
| timestamp | ✓ | ✓ | ✓ | |
| src_ip | ✓ | ✓ | ✓ | Used for IP-based graph edges |
| dst_ip | ✓ | ✓ | ✓ | |
| hostname | ✓ | ✓ | **absent** | Graph converter falls back gracefully |
| username | ✓ | ✓ | **absent** | Graph converter falls back gracefully |
| alert_type | ✓ | ✓ | ✓ | Native TON_IoT attack type — primary label |
| tactic | ✓ | ✓ | ✓ | 50K NaN for normal rows (same pattern) |
| campaign_id | ✓ | ✓ | ✓ | 1-to-1 with alert_type — excluded |
| attack_cat | — | ✓ | **absent** | alert_type fills this role |
| protocol | ✓ | ✓ | ✓ | |
| service | ✓ | ✓ | **absent** | |
| src_bytes | ✓ | ✓ | **absent** | |
| dst_bytes | ✓ | ✓ | **absent** | |
| stage | ✓ | ✓ | **absent** | |
| data_source | ✓ | ✓ | ✓ | |
| src_port | — | — | **extra** | Network feature; available but not in MITRE std |
| dst_port | — | — | **extra** | Network feature |
| label | — | — | **extra** | Binary int (0=normal, 1=attack) |

**Graph construction impact:** The absence of `hostname` and `username` means no host/user entity nodes in the heterogeneous graph. IP-based shared-entity edges still work via `src_ip` (the graph converter checks `src_ip` before `SourceAddress`). V3 smoke-tested successfully on a 500-row sample — 29 clusters, 0% noise.

---

## Label Track Summary

### `alert_type` — 10 classes (0 nulls) — **Recommended primary track**

| Value | Count | Notes |
|-------|-------|-------|
| normal | 50,000 | |
| backdoor | 20,000 | |
| ddos | 20,000 | |
| dos | 20,000 | |
| injection | 20,000 | |
| password | 20,000 | |
| scanning | 20,000 | |
| ransomware | 20,000 | |
| xss | 20,000 | |
| mitm | 1,043 | Smallest class (~49 rows in 10K subset) |

**Usability:** Native TON_IoT label (attack type name). Most informative for clustering. No nulls. 9 of 10 classes are 20K each — highly balanced except mitm. All 10 classes survive a 10K subset.

### `tactic` — 6 attack classes + NaN for Normal (50,000 nulls)

| Value | Count | Notes |
|-------|-------|-------|
| NaN (Normal) | 50,000 | Same NaN pattern as NSL-KDD/UNSW |
| Impact | 60,000 | Largest; covers ddos+dos+ransomware |
| Execution | 40,000 | |
| Persistence | 20,000 | |
| CredentialAccess | 20,000 | |
| Reconnaissance | 20,000 | |
| Collection | 1,043 | Smallest; maps to mitm |

**Usability:** Valid secondary track (7 classes after NaN→"Normal" fill). Coarser than `alert_type` — multiple attack types map to the same tactic (ddos+dos+ransomware → Impact).

### `campaign_id` — 10 classes (0 nulls) — **EXCLUDED**

Perfectly 1-to-1 with `alert_type` (cross-tab confirmed). Evaluating both would produce near-identical metrics with no additional signal.

### `label` — Binary int (0=normal, 1=attack)

| Value | Count |
|-------|-------|
| 0 (normal) | 50,000 |
| 1 (attack) | 161,043 |

**Usability:** Binary; skewed 76.3% attack. `binary_ari` metric requires a binary label. The benchmark handles numeric labels for ARI/AMI — but `benign_label` config in YAML must be set to `0` (not text "normal"). This is a schema difference from NSL-KDD/UNSW — document carefully.

---

## Cross-tabulation Findings

- `alert_type` and `tactic` have clean 1-to-many mapping: each alert_type maps to exactly one tactic, but each tactic covers multiple alert_types (e.g., Impact = ddos+dos+ransomware).
- `campaign_id` is a perfect integer alias for `alert_type` — excludes from benchmark.
- All normal rows (`label=0`) have NaN tactic and `alert_type="normal"`.

---

## Small-Class Warning

| Label track | Class | Count | Expected in 10K subset |
|------------|-------|-------|------------------------|
| `alert_type` / `tactic` | mitm / Collection | 1,043 | ~49 rows |

mitm (1,043 rows, 0.49%) will be sparse but evaluable in a 10K subset (unlike Worms in UNSW at 0.07%). Cluster recall for mitm may still be poor if other clusterers merge it with similar attack types.

---

## Graph Construction Verification

V3 smoke test on 500-row sample: **29 clusters, 0% noise** — pipeline runs without error.
`AlertToGraphConverter` handles missing `hostname`/`username` gracefully (documented in source at lines 488-500: uses `dropna()` if columns absent).

---

## Selected Label Tracks for Benchmark

| Track | Column | Rationale |
|-------|--------|-----------|
| **Primary** | `alert_type` | 10 classes, 0 nulls, native TON_IoT attack type label |
| **Secondary** | `tactic` | 7 classes (NaN→"Normal" fill), mirrors cross-dataset pattern |
| **Binary** | `label` | Binary int (benign_label=0); for binary_ari headline |

Excluded: `campaign_id` (duplicate of alert_type).

---

## Stage 1 Exit Criterion

✅ At least one usable multi-class label track confirmed (`alert_type`, `tactic`).
✅ Label audit written with row count, SHA256, all candidate tracks documented.
✅ Schema differences from NSL-KDD/UNSW documented; graph construction verified.
✅ Decision: 3 benchmark tracks selected, 1 excluded with rationale.
