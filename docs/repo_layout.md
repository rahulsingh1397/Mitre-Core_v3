# MITRE-CORE V3 — Repository Layout

Last updated: 2026-05-19 (post-cleanup baseline)

This page is the first thing to read before adding new code, data, or docs.
Everything has a designated place; when in doubt, check here.

---

## Top-level structure

```
MITRE-CORE-V3/
├── mitre_core/          # V3 library — the only importable package for new code
├── hgnn/                # V2 HGNN core (kept for the validated V3 path; do not add new V2 code)
├── training/            # SSL training entry points (train_graph_mae_v9_multidata_fast.py)
├── utils/               # Minimal utils (only seed_control.py remains; all else archived)
├── benchmark/           # Benchmark harness — run_benchmark.py, YAML specs, results
├── datasets/            # Raw dataset files (large, gitignored at root; loaders in benchmark/)
├── hgnn_checkpoints/    # Model checkpoints (gitignored; canonical ones are the V3 reference)
├── tests/               # All tests — run with: pytest -q
├── docs/                # All documentation
├── scripts/             # Utility scripts (audit scripts, preprocessing)
├── experiments/sweeps/  # Per-dataset sweep scripts (new; future per-dataset sweeps go here)
├── archive/v2_legacy/   # Quarantined V2 code (do not add to; only delete from)
└── checkpoints/         # V3 foundation checkpoint (mitre_core_base_v3.pt)
```

---

## `mitre_core/` — V3 library package

All new research code goes here. Structured as:

```
mitre_core/
├── graph/               # AlertToGraphConverter
├── models/              # MITREHeteroGNN + backbone variants
├── training/            # SSL objectives (NT-Xent, SimCLR, masked-AE, DGI)
├── inference/           # V3CorrelationEngine, EmbeddingConfidenceScorer, ZCA whitening
└── evaluation/          # Unsupervised metrics, benchmark runner, manifest, multi-seed
```

---

## `benchmark/` — Benchmark harness

```
benchmark/
├── run_benchmark.py                  # Single entry point for all benchmark runs
├── datasets_real.yaml                # Per-dataset benchmark config (dev + eval blocks)
├── datasets.yaml                     # Smoke/synthetic benchmark config
├── methods.yaml                      # Method registry (10 methods)
├── clustering_sweep_full_engine.py   # Hyperparameter sweep via full V3 engine (USE THIS)
├── clustering_sweep_standalone.py    # Legacy standalone sweep — DO NOT USE for config selection
├── cache/                            # Cached HGNN embeddings (gitignored)
├── splits/                           # Frozen split index files (.npy + .json)
└── results/
    ├── frozen/<dataset>/v<X.Y>/      # IMMUTABLE — canonical published results
    ├── latest/<dataset>/             # WORKING — current run outputs
    └── history/<dataset>/            # SUPERSEDED — old intermediate results
```

**Rule:** only `frozen/` results are cited in papers and docs. `latest/` is overwritten freely. `history/` is read-only once written.

---

## `docs/` — Documentation

```
docs/
├── plans/
│   ├── MASTER_PLAN_v1.0.md           # Frozen master plan — DO NOT EDIT
│   ├── MASTER_PLAN_v1.0.sha256       # Integrity hash
│   └── README.md                     # Plan index + alignment recheck protocol
├── datasets/
│   ├── _template/                    # Blank template for new datasets
│   ├── nsl_kdd/                      # NSL-KDD: v1.0 FROZEN
│   ├── unsw_nb15/                    # UNSW-NB15: IN PROGRESS
│   └── ...                           # One folder per dataset
├── lessons/                          # Methodology lessons (e.g., Phase 3 sweep failure)
├── architecture.md                   # System architecture
├── benchmark_protocol.md             # Bilot-et-al fair-eval rules
├── reproducibility.md                # REP '25 checklist
├── competitive_analysis.md           # Comparison to MAGIC, FLASH, ORTHRUS
└── archive/v2/                       # Stale V2 docs and reports
```

---

## `tests/` — Test suite

```
tests/
├── test_benchmark_smoke.py           # Full pipeline smoke tests (fast)
├── test_unsupervised_invariant.py    # Asserts no label accessed at inference
├── test_seed_determinism.py          # Reproducibility checks
├── test_split_disjoint.py            # Dev/eval split disjointness (generic)
├── test_frozen_results_have_manifests.py  # Every frozen dir has manifest+summary+config
├── test_master_plan_unchanged.py     # CI enforcement of plan immutability
├── test_nsl_kdd_v1_frozen.py         # Hash integrity + disjoint check for NSL-KDD v1.0
├── test_unsw_nb15_v1_frozen.py       # (created at UNSW freeze)
└── test_package_surface.py           # Smoke imports for mitre_core surface
```

**Markers:**
- `pytest -q` — normal CI; excludes `@pytest.mark.slow`
- `pytest -m slow` — full engine re-run tests; on-demand only

---

## Where new things go

| Thing | Goes in |
|---|---|
| New model variant | `mitre_core/models/` |
| New SSL objective | `mitre_core/training/` |
| New dataset loader | `datasets/loaders/<name>.py` |
| New baseline method | `benchmark/methods/<name>.py` + entry in `methods.yaml` |
| New dataset benchmark config | `benchmark/datasets_real.yaml` |
| New sweep (must use full engine!) | `experiments/sweeps/` |
| New dataset docs | `docs/datasets/<name>/` (copy `_template/`) |
| New methodology lesson | `docs/lessons/` |
| New test | `tests/test_<topic>.py` |
| V2 code you want to resurrect | Read from `archive/v2_legacy/`; port to `mitre_core/` |

---

## What NOT to do

- Do not add new code to `hgnn/` (port it to `mitre_core/` instead)
- Do not add new code to `utils/` (only `seed_control.py` remains by design)
- Do not add new training scripts to `training/` (use `mitre_core/training/`)
- Do not edit `benchmark/results/frozen/` (new results go in `latest/`)
- Do not edit `docs/plans/MASTER_PLAN_v1.0.md` (deviation → new version number)
- Do not run `clustering_sweep_standalone.py` and apply its winner to the full engine
