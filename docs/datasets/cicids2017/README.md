# CICIDS2017 — Dataset Status

**Current version:** IN PROGRESS (Stage 1 — Audit)
**Master plan section:** Part IV.4
**Git tag:** (pending — set at freeze)

## Summary

CICIDS2017 (Canadian Institute for Cybersecurity IDS 2017) is a modern network IDS benchmark
generated in a realistic network environment. It contains 3,119,345 rows across 14 attack types
plus BENIGN traffic. Unlike NSL-KDD and UNSW-NB15 it is stored as parquet, lacks
`hostname`/`username` columns, and is highly imbalanced (BENIGN = 72.9% of rows).
It is the fourth dataset in the benchmark and confirms whether V3's network-IDS claim
extends beyond NSL-KDD/UNSW to a modern dataset with different traffic profiles.

## Headline Numbers (fill at freeze)

| Method | ARI mean ± std (alert_type) | AMI mean ± std (alert_type) |
|---|---|---|
| **MITRE-CORE V3** | TBD | TBD |
| Best baseline | TBD | TBD |

## Files in this folder

| File | Contents |
|---|---|
| [subplan.md](subplan.md) | Active subplan — stage-by-stage execution |
| [audit.md](audit.md) | Label schema audit (Stage 1) |
| [protocol.md](protocol.md) | Split + seed decisions (Stage 2) |
| [decision_log.md](decision_log.md) | Chronological non-trivial choices |
| [learnings.md](learnings.md) | Carry-forward findings for next datasets |
| [v1.0_baseline.md](v1.0_baseline.md) | (created at freeze) Full frozen baseline |
