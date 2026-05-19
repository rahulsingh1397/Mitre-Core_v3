# MITRE-CORE V3 Architecture

MITRE-CORE V3 is organized around a benchmark-first unsupervised alert-correlation stack.

## Principles

- inference is clustering-only
- evaluation is centralized
- benchmark outputs are reproducible artifacts
- semi-unsupervised training is allowed, supervised inference is not

## Core Components

- `mitre_core.inference.correlation_engine.V3CorrelationEngine`
- `mitre_core.evaluation.unsupervised_metrics`
- `benchmark/run_benchmark.py`

## Migration Direction

- reuse authoritative V2 graph and embedding logic
- remove prototype and softmax inference paths from V3-facing APIs
- keep benchmark methods declarative and comparable
