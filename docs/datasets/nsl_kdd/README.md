# NSL-KDD — Dataset Status

**Current version:** v1.0 (FROZEN, 2026-05-19)
**Git tag:** `nsl-kdd-v1.0`
**Master plan section:** Part IV.1

## Summary

NSL-KDD is the first dataset in the MITRE-CORE V3 benchmark. It is fully evaluated and frozen. No further work is planned unless a new checkpoint, new label track, or new baseline set requires a v1.1 re-evaluation.

## Headline Numbers (eval split, tactic track)

| Method | ARI mean ± std | AMI mean ± std |
|---|---|---|
| **MITRE-CORE V3** | **0.602 ± 0.079** | **0.685 ± 0.101** |
| Spectral (raw) | 0.414 ± 0.044 | 0.521 ± 0.063 |
| PCA + K-Means | 0.303 ± 0.021 | 0.442 ± 0.043 |

V3 holds a **+0.19 ARI margin** over the strongest baseline.

## Files in this folder

| File | Contents |
|---|---|
| [v1.0_baseline.md](v1.0_baseline.md) | Full frozen baseline — metrics, hashes, config, known limitations |
| [audit.md](audit.md) | Label schema audit (Phase 2) |
| [protocol.md](protocol.md) | Split + seed decisions |
| [decision_log.md](decision_log.md) | Chronological record of non-trivial choices |
| [learnings.md](learnings.md) | Findings to carry forward to next datasets |

## What NOT to change

- `benchmark/results/frozen/nsl_kdd/v1.0/` — read-only
- `benchmark/splits/nsl_kdd_10000_seed142.npy` — eval split is fixed
- Any result that cites "NSL-KDD v1.0" must match these frozen numbers
