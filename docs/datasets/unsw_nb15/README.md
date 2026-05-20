# UNSW-NB15 — Dataset Status

**Current version:** v1.0 FROZEN
**Master plan section:** Part IV.2
**Git tag:** `unsw-nb15-v1.0`

## Summary

UNSW-NB15 is a synthetic network intrusion detection dataset generated at the Australian Centre for Cyber Security (ACCS). The MITRE-format version contains 175,341 rows across 10 attack categories (Generic, Exploits, Fuzzers, DoS, Reconnaissance, Analysis, Backdoor, Shellcode, Worms) plus Normal traffic. It shares the same 15-column MITRE-format schema as NSL-KDD, making it the closest structural analog for validating that the per-dataset lifecycle transfers.

## Headline Numbers (fill at freeze)

| Method | ARI mean ± std (attack_cat) | AMI mean ± std (attack_cat) |
|---|---|---|
| **MITRE-CORE V3** | **0.564 ± 0.000** | **0.646 ± 0.000** |
| Best baseline (PCA + HDBSCAN) | 0.354 ± 0.000 | 0.614 ± 0.000 |

## Files in this folder

| File | Contents |
|---|---|
| [subplan.md](subplan.md) | Active subplan — stage-by-stage execution |
| [audit.md](audit.md) | Label schema audit (Stage 1) ✅ |
| [protocol.md](protocol.md) | Split + seed decisions (Stage 2) |
| [decision_log.md](decision_log.md) | Chronological non-trivial choices |
| [learnings.md](learnings.md) | Carry-forward findings for next datasets |
| [v1.0_baseline.md](v1.0_baseline.md) | Full frozen baseline ✅ |
