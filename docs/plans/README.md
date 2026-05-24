# MITRE-CORE V3 — Plans Index

## Active Master Plan

**[MASTER_PLAN_v1.1.md](MASTER_PLAN_v1.1.md)** — frozen 2026-05-23
- SHA256: `950BA7BFAF4ECE0389EA663FC9F7568995641DC461A8AEC08CC2BA99CD820234`
- Git tag: `master-plan-v1.1`
- Status: **ACTIVE — DO NOT EDIT IN PLACE**

To deviate from this plan, create `MASTER_PLAN_v1.2.md` with a diff summary at the top.
Update this README to point at the new active version. v1.1 stays immutable.

## Superseded (Immutable)

**[MASTER_PLAN_v1.0.md](MASTER_PLAN_v1.0.md)** — frozen 2026-05-19
- SHA256: `a779e478496b427591e79b2f1808aec48a68c7459d28c0738e61650756a98b54`
- Git tag: `master-plan-v1.0`
- Status: **SUPERSEDED — Read-only. v1.0 frozen artifacts remain immutable.**

## Alignment Recheck Protocol

Before starting any subplan, and before freezing any dataset:
1. Open the active master plan above.
2. Confirm subplan scope, lifecycle stage, and exit criteria match a named section.
3. Cite the section in the subplan's 3-line header.
4. Any deviation → new version number, not an in-place edit.

## Subplans

| Dataset | Subplan | Status |
|---|---|---|
| NSL-KDD | [docs/datasets/nsl_kdd/](../datasets/nsl_kdd/) | v1.0 FROZEN |
| UNSW-NB15 | [docs/datasets/unsw_nb15/subplan.md](../datasets/unsw_nb15/subplan.md) | IN PROGRESS |
| TON_IoT | — | queued |
| CICIDS2017 | — | queued |
| SQTK_SIEM | — | queued |
| DARPA OpTC | — | queued |

## Integrity Check

```bash
pytest tests/test_master_plan_unchanged.py
```
