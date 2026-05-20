# TON_IoT — Dataset Status

**Current version:** IN PROGRESS (Stage 2 — Protocol Freeze)
**Master plan section:** Part IV.3
**Git tag:** (pending — set at freeze)

## Summary

TON_IoT (Things of Network IoT) is an IoT/IIoT network dataset generated at the UNSW Cyber Range. It contains 211,043 rows across 9 attack types (backdoor, ddos, dos, injection, password, scanning, ransomware, xss, mitm) plus Normal traffic. Unlike NSL-KDD and UNSW-NB15, it is stored as parquet and lacks `hostname`/`username` columns (replaced by `src_port`/`dst_port`/`label`). It is the first non-network-IDS dataset in the benchmark and tests whether V3's zero-shot claim survives a domain shift to IoT traffic.

## Headline Numbers (fill at freeze)

| Method | ARI mean ± std (alert_type) | AMI mean ± std (alert_type) |
|---|---|---|
| **MITRE-CORE V3** | TBD | TBD |
| Best baseline | TBD | TBD |

## Files in this folder

| File | Contents |
|---|---|
| [subplan.md](subplan.md) | Active subplan — stage-by-stage execution |
| [audit.md](audit.md) | Label schema audit (Stage 1) ✅ |
| [protocol.md](protocol.md) | Split + seed decisions (Stage 2) |
| [decision_log.md](decision_log.md) | Chronological non-trivial choices |
| [learnings.md](learnings.md) | Carry-forward findings for next datasets |
| [v1.0_baseline.md](v1.0_baseline.md) | (created at freeze) Full frozen baseline |
