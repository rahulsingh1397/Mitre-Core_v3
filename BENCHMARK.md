# MITRE-CORE V3 Benchmark Protocol

MITRE-CORE V3 evaluates unsupervised and semi-unsupervised alert-correlation methods under a single benchmark harness.

## Scope

- Zero labels at inference time
- Semi-unsupervised training allowed
- Shared dataset splits, seeds, metrics, and hardware reporting
- Negative results retained in committed benchmark outputs

## Default Seeds

- 42
- 43
- 44

## Benchmark Datasets

- NSL-KDD
- UNSW-NB15
- TON_IoT
- DARPA OpTC
- SQTK_SIEM
- CICIDS2017
- DARPA TC3
- NODLINK eval

## Core Rules

1. Methods must expose inference without requiring labels.
2. Evaluation code owns all metric computation.
3. Result CSVs must include dataset, method, seed, metrics, latency, and memory fields.
4. Synthetic smoke benchmark must run in CI.

## Reported Metrics

- AMI
- ARI
- binary_ARI
- purity
- silhouette_cosine
- attack_f1
- cluster_attribution_f1
- latency_seconds_per_10k
- peak_gpu_gb
