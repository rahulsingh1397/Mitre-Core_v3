# MITRE-CORE V3 Improvement Log

## Baseline Assessment

### 2026-05-17 — End-to-End Benchmark Sanity Run
- **Command**: `python -m benchmark.run_benchmark --output benchmark/results/e2e_benchmark_results.csv`
- **Status**: Completed
- **Scope**: Current synthetic smoke benchmark only
- **Key finding**: The current V3 benchmark harness validates pipeline plumbing, metric generation, and output artifacts, but it is not yet a real-dataset benchmark.
- **Important caveat**: `MITRE-CORE V3` in the current harness is still a placeholder path using synthetic features and a `KMeans` prediction path rather than full HGNN inference on benchmark datasets.
- **Artifacts**:
  - `benchmark/results/e2e_benchmark_results.csv`
  - `benchmark/results/e2e_benchmark_results_summary.csv`

## Dataset Improvement Template

For each dataset pass, record the following:

### Dataset: `<name>`
- **Date**:
- **Owner**:
- **Objective**:
- **Dataset version / source**:
- **Split / evaluation protocol**:
- **Methods compared**:
- **Checkpoint(s)**:
- **Seeds**:
- **Metrics**:
- **Latency / memory**:
- **Result summary**:
- **Observed failure mode**:
- **Changes made**:
- **Files touched**:
- **Artifacts generated**:
- **Decision**:
- **Next action**:

## Current Priority Order
1. Build real-dataset benchmark wiring before claiming benchmark quality.
2. Start with the dataset that best matches the canonical unsupervised checkpoint to establish a trustworthy baseline.
3. Improve one dataset at a time with frozen seeds, fixed splits, and tracked artifacts.

### Dataset: `NSL-KDD`
- **Date**: 2026-05-17
- **Owner**: Cascade
- **Objective**: Establish the first real V3 benchmark baseline on a dataset aligned with the canonical unsupervised checkpoint.
- **Dataset version / source**: `datasets/nsl_kdd/mitre_format.csv`
- **Split / evaluation protocol**: 10,000-row stratified sample per seed from MITRE-format CSV.
- **Methods compared**:
  - `K-Means (raw)`
  - `DBSCAN (raw)`
  - `HDBSCAN (raw)`
  - `MITRE-CORE V3`
- **Checkpoint(s)**: `hgnn_checkpoints/network_v9_v3/network_it_best.pt`
- **Seeds**: `42, 43, 44`
- **Metrics**:
  - `MITRE-CORE V3`: `AMI mean=0.6973`, `ARI mean=0.6818`, `purity mean=0.9804`
  - `K-Means (raw)`: `ARI mean≈0.0001`
  - `DBSCAN (raw)`: `ARI mean=0.0000`
  - `HDBSCAN (raw)`: `ARI mean=0.0051`
- **Latency / memory**:
  - `MITRE-CORE V3`: `10.90s / 10k` on CPU benchmark path
  - `peak_gpu_gb`: `0.0` in current harness
- **Result summary**: The new V3 real-dataset benchmark lane is working. On the first real dataset, MITRE-CORE V3 is dramatically stronger than raw-feature baselines, but the current protocol shows high variance across seeds because the sampled evaluation subset changes by seed.
- **Observed failure mode**: The benchmark is currently mixing two sources of variance: model/clustering seed variance and data sampling variance. This makes the protocol weaker than current artifact and fair-evaluation standards.
- **Changes made**:
  - Added real-dataset support to `mitre_core/evaluation/benchmark.py`
  - Added benign-label-aware metric support
  - Added `benchmark/datasets_real.yaml`
  - Added `make benchmark-real`
  - Fixed module-based benchmark entrypoints in repo surfaces
- **Files touched**:
  - `mitre_core/evaluation/benchmark.py`
  - `mitre_core/evaluation/unsupervised_metrics.py`
  - `benchmark/datasets_real.yaml`
  - `Makefile`
  - `Dockerfile`
  - `docker-compose.yml`
- **Artifacts generated**:
  - `benchmark/results/benchmark_real_results.csv`
  - `benchmark/results/benchmark_real_results_summary.csv`
- **Decision**: Do not tune the model yet. First stabilize the evaluation protocol by freezing the NSL-KDD benchmark subset or running the full deterministic dataset path.
- **Next action**: Freeze NSL-KDD sampling so per-seed comparisons isolate clustering/model variance instead of data drift.

### Dataset: `NSL-KDD` — Protocol Stabilization Step
- **Date**: 2026-05-17
- **Objective**: Freeze the sampled benchmark subset so multi-seed reporting reflects method variance instead of evaluation-set drift.
- **Change**: Added dataset-level `sample_seed` support to the V3 benchmark harness and fixed `NSL-KDD` to `sample_seed: 42` in `benchmark/datasets_real.yaml`.
- **Expected effect**: Lower variance across seeds and a stronger match to artifact-evaluation and fair-benchmark norms.

### Dataset: `NSL-KDD` — Phase 1 Protocol Hardening Verification
- **Date**: 2026-05-18
- **Objective**: Verify persisted split indices, manifest emission, and disjoint dev/eval split support end to end.
- **Verification coverage**:
  - `tests/test_split_disjoint.py`
  - `tests/test_benchmark_smoke.py`
- **Checks run**:
  - `python -m pytest tests/test_split_disjoint.py tests/test_benchmark_smoke.py -q`
  - `python -m benchmark.run_benchmark --methods benchmark/methods.yaml --datasets benchmark/datasets_real.yaml --output benchmark/results/benchmark_real_results.csv`
- **Artifacts generated**:
  - `benchmark/results/benchmark_real_results.csv`
  - `benchmark/results/benchmark_real_results_summary.csv`
  - `benchmark/results/benchmark_real_results_manifest.json`
  - `benchmark/splits/nsl_kdd_10000_seed42.npy`
  - `benchmark/splits/nsl_kdd_10000_seed42.json`
  - `benchmark/splits/nsl_kdd_10000_seed142.npy`
  - `benchmark/splits/nsl_kdd_10000_seed142.json`
- **Result summary**:
  - Manifest now records dataset SHA256, checkpoint SHA256, split file SHA256, environment details, command provenance, and excluded split paths.
  - The persisted eval split for `NSL-KDD` was reused across benchmark seeds, producing zero metric variance for methods whose outputs are deterministic on the frozen subset.
  - Disjoint dev/eval split behavior is covered by test and backed by persisted split artifacts for seeds `42` and `142`.
- **Decision**: Phase 1 protocol hardening is complete and verified.
- **Next action**: Proceed to Phase 2 label-track validation on the hardened NSL-KDD protocol.

### Dataset: `NSL-KDD` — Phase 2 Label-Track Validation
- **Date**: 2026-05-18
- **Objective**: Validate NSL-KDD label tracks, choose meaningful secondary tracks, and extend the benchmark to report per-track metrics.
- **Label audit on frozen eval split**:
  - `tactic`: 8 classes (current primary)
  - `alert_type`: binary (normal/attack) — independent, useful for binary validation
  - `campaign_id`: 15 classes — largely independent from tactic, good alternate multi-class track
  - `stage`: 6 classes but perfectly redundant with `alert_type` (`Normal` = normal, stages = attack)
- **Evaluation contract decided**:
  - Primary: `tactic` (kept)
  - Secondary: `alert_type` (binary), `campaign_id` (multi-class alternate)
  - Deferred: `stage` (redundant with alert_type)
- **Implementation**:
  - Added `alt_label_cols` support to `mitre_core/evaluation/benchmark.py`
  - Extended `LoadedDataset` with `label_tracks` dictionary
  - Updated `run_benchmark` to loop over all configured tracks and emit a `label_track` column
  - Updated `benchmark/datasets_real.yaml` with `alt_label_cols: [alert_type, campaign_id]`
  - Added `test_multi_track_label_evaluation` to `tests/test_benchmark_smoke.py`
- **Verification**:
  - `python -m pytest tests/test_benchmark_smoke.py -q` → 3 passed
  - Real benchmark rerun completed successfully with multi-track results
- **Artifacts generated**:
  - `benchmark/results/benchmark_real_results.csv` (now includes `label_track` column)
  - `benchmark/results/benchmark_real_results_summary.csv`
  - `benchmark/results/benchmark_real_results_manifest.json`
- **Key NSL-KDD results on frozen eval split (MITRE-CORE V3, seed 42)**:
  - `tactic` track: AMI=0.714, ARI=0.632, attack_f1=1.0, purity=0.991
  - `alert_type` track: AMI=0.557, ARI=0.501, attack_f1=0.991, binary_ari=0.967
  - `campaign_id` track: AMI=0.784, ARI=0.675, attack_f1=1.0, purity=0.968
- **Decision**: Phase 2 label-track validation is complete and verified. The benchmark now supports multi-track evaluation.
- **Next action**: Proceed to Phase 3 clustering hyperparameter sweep on cached NSL-KDD embeddings.

### Dataset: `NSL-KDD` — Phase 3 Clustering Hyperparameter Sweep
- **Date**: 2026-05-19
- **Objective**: Extract dev-split embeddings once, run a controlled HDBSCAN clustering sweep on cached embeddings, select winner on dev, and validate on eval.
- **Implementation**:
  - Created `benchmark/clustering_sweep.py` with cache-first embedding extraction
  - Swept 36 HDBSCAN configs on the frozen dev split (seed 42):
    - `hdbscan_min_cluster_size`: [3, 5, 10]
    - `hdbscan_pca_components`: [8, 16, 24]
    - `hdbscan_cluster_selection_epsilon`: [0.0, 0.05, 0.1, 0.15]
- **Sweep winner on dev** (standalone HDBSCAN on cached embeddings):
  - `min_cluster_size=10`, `pca_components=8`, `epsilon=0.0`
  - primary_ari=0.3377, primary_ami=0.5142, n_clusters=29
- **Full-engine eval with winner config**:
  - Applied winner to `benchmark/datasets_real.yaml` and reran eval
  - **Result**: significant degradation vs original config
    - `tactic` track ARI dropped from 0.632 → 0.078
    - `alert_type` track ARI dropped from 0.501 → 0.026
    - `campaign_id` track ARI dropped from 0.675 → 0.090
- **Root cause**: The standalone HDBSCAN sweep on raw cached embeddings does not replicate the full `EmbeddingConfidenceScorer` clustering pipeline used by the V3 engine (GAEC confidence scoring, raw-feature concatenation, etc.). Therefore, the standalone winner does not translate to the full engine.
- **Decision**: Reverted `datasets_real.yaml` to the original well-tuned config:
  - `hdbscan_min_cluster_size=5`, `hdbscan_pca_components=16`, `hdbscan_cluster_selection_epsilon=0.1`
- **Artifacts generated**:
  - `benchmark/results/nsl_kdd_clustering_sweep.csv`
  - `benchmark/results/benchmark_real_results_phase3.csv` (degraded run, retained for reference)
  - Cached embeddings: `benchmark/cache/embeddings/e818cffc233b31e8.npy`
- **Verification**:
  - Added `test_clustering_sweep_caches_embeddings_and_selects_winner` to `tests/test_benchmark_smoke.py`
  - `python -m pytest tests/test_benchmark_smoke.py -q` → 4 passed
- **Key learning**: Isolating clustering from representation is useful for understanding sensitivity, but winner configs from simplified standalone clustering may not transfer to the full production pipeline. Future sweeps should either (a) use the full engine for each config, or (b) replicate the `EmbeddingConfidenceScorer` clustering logic exactly.
- **Decision**: Phase 3 is complete with the documented learning. The original clustering config remains locked as the NSL-KDD benchmark standard.
- **Next action**: Proceed to Phase 4 stronger unsupervised baselines.

### Dataset: `NSL-KDD` — Phase 4 Stronger Unsupervised Baselines
- **Date**: 2026-05-19
- **Objective**: Add stronger unsupervised baselines to the benchmark harness to produce an honest comparison ladder for NSL-KDD.
- **Implementation**:
  - Extended `mitre_core/evaluation/benchmark.py` `_run_baseline` with three embedding-based baselines:
    - `kmeans_emb` — K-Means on HGNN embeddings
    - `spectral_emb` — Spectral Clustering on HGNN embeddings
    - `hdbscan_emb` — HDBSCAN on HGNN embeddings
  - Updated `run_benchmark` to lazily extract HGNN embeddings once per seed and reuse them across embedding-based baselines and the V3 method.
  - Enabled the new baselines in `benchmark/methods.yaml`.
  - Updated smoke-test assertion to include the new methods.
- **Verification**:
  - `python -m pytest tests/test_benchmark_smoke.py -q` → 4 passed
  - Real NSL-KDD benchmark rerun with all baselines completed successfully.
- **Artifacts generated**:
  - `benchmark/results/benchmark_real_results_phase4.csv`
  - `benchmark/results/benchmark_real_results_phase4_summary.csv`
- **NSL-KDD comparison ladder on frozen eval split (primary `tactic` track, mean ARI across seeds 42–44)**:

| Method | ARI | AMI | purity | attack_f1 |
|--------|-----|-----|--------|-----------|
| K-Means (raw) | -0.0001 | -0.0001 | 0.535 | 0.667 |
| DBSCAN (raw) | 0.000 | 0.000 | 0.535 | 0.667 |
| HDBSCAN (raw) | 0.005 | 0.001 | 0.580 | 0.840 |
| **K-Means (emb)** | **0.189** | **0.400** | **0.912** | **0.996** |
| **Spectral (emb)** | **0.218** | **0.481** | **0.942** | **0.998** |
| **HDBSCAN (emb)** | **0.039** | **0.255** | **0.928** | **0.977** |
| **MITRE-CORE V3** | **0.602** | **0.685** | **0.984** | **0.997** |

- **Key findings**:
  - Raw-feature baselines are essentially at chance on NSL-KDD.
  - Embedding-based baselines dramatically improve over raw baselines, confirming that the HGNN backbone produces meaningful representations.
  - Spectral (emb) is the strongest embedding-only baseline (ARI=0.218), but still ~0.38 ARI below the full MITRE-CORE V3 pipeline.
  - MITRE-CORE V3 maintains a clear margin over all baselines, validating that the confidence-scored clustering and full pipeline add genuine value beyond simple reclustering of embeddings.
- **Decision**: Phase 4 is complete. The benchmark now includes a meaningful unsupervised comparison ladder.
- **Next action**: Proceed to Phase 5 (reporting refinement, graph ablations, or any remaining cleanup).
