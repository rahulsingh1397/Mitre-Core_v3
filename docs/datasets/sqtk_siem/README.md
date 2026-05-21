# SQTK_SIEM — Dataset Status

**Current version:** IN PROGRESS
**Master plan section:** Part IV.5
**Git tag:** (pending — set at freeze)

## Summary

SQTK_SIEM is a heterogeneous SIEM dataset with 5,100 alert rows. It is the smallest dataset in the benchmark and the first to use a dataset-specific checkpoint (`siem_supcon_v4/best.pt`), testing whether a SIEM-trained encoder outperforms the generic `network_v9_v3` checkpoint on SIEM data.

## Headline Numbers (fill at freeze)

| Method | ARI mean ± std | AMI mean ± std |
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
