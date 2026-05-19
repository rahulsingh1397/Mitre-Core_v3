# MITRE-CORE V3

MITRE-CORE V3 is an unsupervised and semi-unsupervised alert-correlation repository focused on fair evaluation, reproducible benchmarking, and benchmark-first engineering.

## V3 Scope

- unsupervised inference only
- semi-unsupervised training allowed
- benchmark-first repository layout
- repo-only checkpoint metadata
- honest reporting of negative results

## Current V3 Status

This repository now includes the first V3 scaffold:

- `pyproject.toml`
- `BENCHMARK.md`
- `benchmark/run_benchmark.py`
- `benchmark/methods.yaml`
- `benchmark/datasets.yaml`
- `mitre_core/` package with a V3 inference wrapper
- unsupervised invariant and smoke benchmark tests
- CI workflows for tests and benchmark smoke runs

## Research-Informed Evaluation Direction

Recent provenance and intrusion-detection benchmark work emphasizes:

- same splits across methods
- same tuning budget across methods
- instability reporting across seeds
- strong simple baselines alongside heavyweight graph methods
- practicality metrics such as latency and memory

MITRE-CORE V3 adopts that direction explicitly.

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .[dev]
make benchmark
make test
```

## Benchmark Layout

- `benchmark/` — benchmark entrypoint, dataset spec, method roster, results, figures
- `mitre_core/evaluation/` — centralized metric computation and multi-seed aggregation
- `mitre_core/inference/` — V3-safe inference wrapper
- `tests/test_unsupervised_invariant.py` — blocks label-dependent inference paths
- `tests/test_benchmark_smoke.py` — CI smoke benchmark

## V3 Guardrails

- `HGNNCorrelationEngine` now rejects `pure_unsupervised=False`
- prototype inference is rejected from the V3-facing path
- benchmark evaluation rejects inference functions that require label arguments

## Planned Migration Work

- port authoritative graph construction into `mitre_core/graph/`
- port unsupervised-safe correlation logic into `mitre_core/inference/`
- refactor baseline experiment scripts into benchmark modules
- expand the benchmark roster from scaffolded methods to the full V3 suite

## License

MIT License
