# TON_IoT — Dataset Status

**Current version:** v1.0 FROZEN — 2026-05-20
**Master plan section:** Part IV.3
**Git tag:** `ton-iot-v1.0`

## Summary

TON_IoT (Things of Network IoT) is an IoT/IIoT network dataset generated at the UNSW Cyber Range. It contains 211,043 rows across 9 attack types (backdoor, ddos, dos, injection, password, scanning, ransomware, xss, mitm) plus Normal traffic. Unlike NSL-KDD and UNSW-NB15, it is stored as parquet and lacks `hostname`/`username` columns (replaced by `src_port`/`dst_port`/`label`). It is the first non-network-IDS dataset in the benchmark and tests whether V3's zero-shot claim survives a domain shift to IoT traffic.

**Headline result:** V3 does **not** dominate on TON_IoT. K-Means (raw) wins by +0.199 ARI. This is a genuine finding — the zero-shot claim holds for network IDS (NSL-KDD, UNSW-NB15) but not IoT traffic with the current checkpoint.

## Headline Numbers (v1.0, alert_type track, seeds 42/43/44)

| Method | ARI mean ± std | AMI mean ± std | Notes |
|---|---|---|---|
| K-Means (raw) | **0.622 ± 0.033** | **0.769 ± 0.023** | n_clusters=10 (true class count — privileged prior) |
| K-Means (emb) | 0.612 ± 0.014 | 0.784 ± 0.008 | |
| PCA + K-Means | 0.507 ± 0.034 | 0.728 ± 0.017 | |
| **MITRE-CORE V3** | **0.423 ± 0.000** | **0.705 ± 0.000** | 55 clusters predicted (HDBSCAN over-segmentation) |
| PCA + HDBSCAN | 0.248 ± 0.000 | 0.621 ± 0.000 | |

**V3 margin over best baseline: −0.199 ARI (Path B)**

Swept mcs=300 improves V3 to ARI=0.474 on eval — gap to K-Means(raw) reduces to 0.148, still above the honest cap. Default config frozen for cross-dataset consistency.

## Files in this folder

| File | Contents |
|---|---|
| [subplan.md](subplan.md) | Subplan — all stages ✅ |
| [audit.md](audit.md) | Label schema audit (Stage 1) ✅ |
| [protocol.md](protocol.md) | Split + seed decisions (Stage 2) ✅ |
| [investigation.md](investigation.md) | Path B root-cause analysis + sweep results (Stage 4) ✅ |
| [decision_log.md](decision_log.md) | Chronological non-trivial choices ✅ |
| [learnings.md](learnings.md) | Carry-forward findings for next datasets ✅ |
| [v1.0_baseline.md](v1.0_baseline.md) | Full frozen baseline ✅ |
