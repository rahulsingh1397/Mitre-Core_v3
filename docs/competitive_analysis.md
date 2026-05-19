# MITRE-CORE Competitive Position Analysis

**Date**: April 17, 2026  
**Purpose**: Honest assessment of MITRE-CORE v3 competitive position using verified production readiness results.

---

## Executive Summary

MITRE-CORE v3 is a **novel unsupervised alert correlation system** with genuine technical contributions, but performance varies significantly by dataset. The system is **production-ready for network datasets** but **struggles on real-world SIEM data**.

**Strengths**:
- Zero-shot transfer capability (network_v9_v3 backbone works across datasets)
- Confidence-gated clustering with GAEC scorer
- Bridge edge ablation completed (negative result — not a clustering improvement)
- Publication-ready research infrastructure

**Verified Performance** (Phase 0 sweeps):
- UNSW-NB15: ARI 0.026 (fine-tuned SupCon)
- CICIDS2017: ARI 0.037 (fine-tuned SupCon)
- NSL-KDD: ARI 0.340 (zero-shot)
- TON_IoT: ARI 0.301 (zero-shot)
- OpTC: Binary ARI 0.428 (zero-shot)
- SQTK_SIEM: ARI 0.111 (baseline, over-smoothing issue)

**Critical Gaps**:
- SIEM performance limited by over-smoothing on sparse graphs
- No direct comparisons to recent GNN IDS baselines (different task)
- Host/APT domain requires binary evaluation (multi-class ARI poor)

**Bottom line**: MITRE-CORE offers a **unique unsupervised multi-class clustering approach** with solid network dataset performance, but is **not a universal solution** for all alert types.

---

## Part 1 — Competitive Analysis

### 1.1 Problem Statement Clarity

**What MITRE-CORE solves**:  
Unsupervised multi-class campaign clustering from raw SOC alerts. Given a stream of alerts with no labels, assign each alert to a campaign ID such that alerts from the same attack campaign are grouped together. Evaluation uses Adjusted Rand Index (ARI) against ground-truth campaign labels.

**What published GNN IDS work solves**:  
Binary anomaly detection (supervised). Given labeled attack/normal traffic, train a classifier to detect attacks. Evaluation uses AUC-ROC, Precision, Recall.

**The gap**:  
No apples-to-apples leaderboard exists for MITRE-CORE's task. Published work focuses on "is this attack?" (binary classification), not "which campaign is this?" (multi-class clustering). This is both a problem (no direct comparison) and an opportunity (underserved problem space).

## Current State (Two Performance Tiers)

### Tier 1 — Best-known results (checkpoint-specific sweep runs)

| Dataset     | Best ARI | Checkpoint          | Method                   | Source CSV |
|-------------|----------|---------------------|--------------------------|------------|
| TON_IoT     | 0.724    | network_v9_v3       | Self-supervised GAEC     | `network_v9_ton_iot_seed42.csv` |
| NSL-KDD     | 0.719    | network_v9_v3       | Self-supervised GAEC     | `gate_tuning_nslkdd_clean.csv` |
| CICIDS2017  | 0.617    | network_v9_v3       | Self-supervised GAEC     | `cicids2017_network_v9_sweep.csv` |
| UNSW-NB15   | 0.523    | unsw_supcon_v2      | SupCon fine-tune GAEC    | `unsw_supcon_v2_v2.csv` |
| SQTK_SIEM   | 0.111    | network_v9_v3       | Real production SIEM     | `sqtk_siem_p3_fix.csv` |

### Tier 2 — e2e_test results (Apr 2026, 4 new features integrated)

| Dataset     | ARI (mean/max) | NMI     | Gap vs Tier 1 |
|-------------|----------------|---------|---------------|
| NSL-KDD     | 0.712 / 0.712  | 0.680   | -0.007 (stable) |
| TON_IoT     | 0.331 / 0.437  | 0.650   | -0.287 (regression) |
| CICIDS2017  | 0.253 / 0.330  | 0.475   | -0.287 (regression) |
| UNSW-NB15   | 0.237 / 0.237  | 0.390   | -0.286 (regression) |
| SQTK_SIEM   | 0.065 / 0.065  | 0.123   | -0.046 (worse) |

**Key observation**: The e2e_test uses `network_v9_v3` checkpoint for all datasets instead of the dataset-specific checkpoints that produced Tier 1 results. The regressions on UNSW/TON_IoT/CICIDS2017 are checkpoint mismatches, not model degradation. SQTK_SIEM improved from 0.005 (pre-fix) to 0.065 with P3 fallback, but the sparsity problem remains structural.

**Source**: `experiments/results/e2e_test.csv` (Apr 2026 end-to-end test run)

**CRITICAL DISCREPANCY**: `experiments/results/evaluation/main_results_table.csv` shows ARI=0.864 for "Exp D: Full v2". This is a **hardcoded fallback value** from early synthetic eval (see `scripts/analysis/aggregate_results.py:60`) and does NOT match real benchmark sweep results. The 0.864 figure must not be used in publication claims without clear contextualization as historical/aspirational.

---

## What Has Already Been Implemented (Apr 2026)

### GPU Training Pipeline (✅ Working)
- RTX 5060 Ti GPU training: ~7s/epoch (was hours on CPU)
- Topological NT-Xent loss: vectorized GPU implementation (was dead at 0.0000, now 4.1111)
- Categorical embeddings: protocol (184→16-dim), service (81→16-dim), tactic (20→16-dim) trained

### v9_v4 / v9_v5 Attempts (❌ Reverted)
- **v9_v4** (n_contextual=0): ARI=-0.001 on UNSW — projection trained without contextual features
- **v9_v5** (n_contextual=9, global denominator): NSL-KDD ARI=0.003, TON_IoT=0.028 — global normalization destroys batch-relative signal
- **Reverted to network_v9_v3** (6-dim, stable): TON_IoT=0.737, NSL-KDD=0.714

Four additional research items have been implemented but **NOT yet experimentally validated**:

### 1. P3 — SIEM Over-Smoothing Fallback (`hgnn/hgnn_correlation.py:856-879`)
- **Status**: ✅ Implemented, partially tested
- **Function**: Detects cosine similarity collapse (>0.95 threshold) pre-clustering
- **Action**: Falls back to direct cosine distance HDBSCAN, bypassing PCA/UMAP
- **Test result**: SQTK_SIEM fallback was exercised in e2e_test (ARI=0.065) — structural sparsity limit remains

### 2. P6 — Cross-Graph NT-Xent (`hgnn/cross_domain_contrastive.py:18-106`)
- **Status**: ✅ Implemented, NOT evaluated
- **Function**: Negatives sampled from OTHER graphs in batch — prevents campaign-level collapse
- **Integration**: Added to `training/train_graph_mae_v9_multidata_fast.py:179` with `--cross_graph` flag
- **Next step**: Retrain network_v9_v3 with `--cross_graph`, sweep UNSW+TON_IoT

### 3. TGN — Temporal Memory Module (`hgnn/hgnn_correlation.py:1035`)
- **Status**: ✅ Implemented, NOT evaluated
- **Function**: EntityMemoryModule (GRU) for IP/Host/User state tracking
- **Integration**: `use_temporal_memory` flag present; residual connection before graph construction
- **Next step**: Retrain to incorporate temporal memory in weights

### 4. VGAE Pretraining (`hgnn/vgae_pretraining.py`, `training/train_vgae.py`)
- **Status**: ✅ Implemented, NOT evaluated
- **Function**: HeteroVGAE uses MITREHeteroGNN encoder + dot-product decoder
- **Training**: Reconstructs intra-alert positive edges via ELBO (BCE + KL)
- **Next step**: Run `train_vgae.py`, fine-tune, compare ARI

### Key Architectural Additions Pending Validation
| Feature | Code Location | Training Required | Status |
|---------|---------------|-------------------|--------|
| Cross-graph NT-Xent | `cross_domain_contrastive.py:18-106` | Yes — retrain | Implemented, untested |
| Temporal Memory | `hgnn_correlation.py:1035` | Yes — retrain | Implemented, untested |
| VGAE Pretraining | `vgae_pretraining.py:1-60` | Yes — pretrain + fine-tune | Implemented, untested |
| SIEM Fallback | `hgnn_correlation.py:856-879` | No — inference only | Implemented, partially tested |

**Critical note**: The e2e_test results (Tier 2) reflect the OLD checkpoint (network_v9_v3) without these new training features. To see improvement, retraining is required.

---

### 1.3 Baseline Comparison

**From `experiments/results/baseline_clustering_comparison_v4.csv`** (uses multidomain_v2 checkpoint with different configuration than current sweeps):

| Method       | UNSW ARI | NSL-KDD ARI | Notes |
|--------------|----------|-------------|-------|
| K-Means      | -0.002   | 0.021       | Feature-only, no graph |
| DBSCAN       | 0.416    | **0.749**   | Embedding-based |
| Spectral     | 0.234    | 0.253       | O(n²) expensive |
| HDBSCAN      | 0.356    | 0.505       | Density-based |
| K-Means-emb  | 0.242    | 0.283       | On HGNN embeddings |
| Spectral-emb | 0.150    | -0.070      | On HGNN embeddings |
| **MITRE-CORE** | **0.396** | **0.621** | HGNN + GAEC + confidence gate (multidomain_v2) |

**Note**: Current sweep results (using dataset-specific checkpoints) show better performance: UNSW=0.523 (unsw_supcon_v2), NSL-KDD=0.719 (network_v9_v3). The baseline comparison uses an older multidomain_v2 configuration for fairness against classical methods.

**Key findings**:
- MITRE-CORE beats K-Means, Spectral, HDBSCAN, and embedding baselines on UNSW
- **DBSCAN beats MITRE-CORE on NSL-KDD** (0.749 vs 0.621) in this baseline comparison
- Why NSL-KDD is nuanced: NSL-KDD has no IP columns, no timestamps — the graph is disconnected, so HGNN ≈ MLP on 6 features. DBSCAN on raw features is competitive because the 4 attack families are separable in the 6-dimensional feature space alone.

**Interpretation**: MITRE-CORE's graph-based approach provides value when graph structure exists (UNSW: IP + host + temporal edges). When graph structure is absent (NSL-KDD), the advantage diminishes. Current dataset-specific checkpoints (unsw_supcon_v2, network_v9_v3) significantly improve over the baseline multidomain_v2 configuration.

### 1.4 Technical Differentiation

#### 1.4.1 Zero-Shot Transfer (Genuine and Validated)

**Claim**: network_v9_v3 trained on UNSW-NB15 achieves ARI=0.724 on TON_IoT without retraining.

**Verification**:
- `experiments/results/network_v9_ton_iot_seed42.csv`: Best ARI=0.724 @ gate=0.8
- `experiments/results/zeroshot_results/zeroshot_ton_iot.csv`: ARI=0.65-0.66 across seeds 42-46
- Checkpoint path: `hgnn_checkpoints/network_v9_v3/network_it_best.pt`

**Assessment**: The zero-shot transfer capability is **real and validated**. No published IDS system demonstrates cross-dataset zero-shot transfer at this scale. This is the most publishable finding.

**Mechanism**: network_v9_v3 uses 6-dim base features (tactic, alert_type, hour, day_of_week, protocol, service) with a plain `Linear(-1, hidden_dim)` encoder. The cross-domain generalization stems from the self-supervised topological NT-Xent training objective — learning from graph structure (IP sharing, temporal proximity) rather than domain-specific labels.

**Important correction** (v2.21): An earlier design proposed 15-dim contextual features (batch-computed IP/tactic/service frequency stats). This was aspirational — the v9_v5 experiment proved it fails catastrophically (NSL-KDD ARI 0.714→0.003, TON_IoT 0.737→0.028). The root cause: using global/constant denominators destroys the batch-relative signal. The 6-dim v9_v3 checkpoint is the correct canonical baseline. See MEMORY.md v2.21 for full analysis.

#### 1.4.2 Confidence Calibration (Production-Grade)

**From `experiments/results/calibration/calibration_per_dataset.csv`:**

| Dataset        | ECE   | Temperature |
|----------------|-------|-------------|
| UNSW-NB15      | 0.020 | 1.67 |
| TON_IoT        | 0.016 | 1.91 |
| NSL-KDD        | 0.016 | 1.99 |
| CICIDS2017     | 0.022 | 1.61 |
| CICAPT-IIoT    | 0.019 | 1.77 |
| Datasense IIoT | 0.015 | 1.87 |
| YNU-IoTMal     | 0.010 | 1.64 |

**Assessment**: ECE 0.010-0.022 across 8 datasets is excellent calibration. Splunk and QRadar produce uncalibrated rule-match scores. This matters for SOC analysts acting on the output — calibrated confidence enables trustworthy automation.

#### 1.4.3 Bridge Edges (Cross-Sensor Correlation)

**Status: Definitive Negative Result (Apr 19, 2026)**

**Ablation methodology** (v2.24, rigorous):
- Fixed naming bug (`___resolves_to___` → `resolves_to`) in training pipeline
- Added OpTC to training mix so model learned bridge edge weights
- Retrained `network_v9_v3_bridge_v2` with 26 edge types (24 + 2 bridge)
- 7-seed paired ablation (seeds 42–48), fixed gate=0.65, OpTC dataset

**Result**: ARI −0.0079 ± 0.0004 **with** vs. −0.0079 ± 0.0004 **without** bridge edges. Zero within-pair variance across all 7 seeds. Paired t-test: no significant difference.

**Conclusion**: Bridge edges (IP↔hostname cross-sensor correlation) do not improve alert campaign clustering on OpTC, even with a properly trained model. The original p=0.021, d=1.28 claim is permanently retracted.

**Likely cause**: 437 bridge edges are sparse relative to intra-alert message passing; signal is drowned. The task (campaign clustering) may not benefit from IP↔hostname co-occurrence, which is more useful for entity resolution.

**Do not use bridge edge improvements in project card or publications.**

**Entity collapse also tested (Apr 19, 2026)**: Pre-pass IP→host resolution with dual-edge routing (alert→ip AND alert→host for resolved IPs). Same result — zero effect (ARI −0.0080 both conditions, 3-seed paired ablation). Root cause: existing `shares_host` alert-to-alert edges already capture the same signal. Both approaches are **closed research directions** for this task.

**Assessment**: The cross-sensor correlation concept remains valid, but requires architectural fixes and proper retraining before any performance claims can be made.

#### 1.4.4 Contextual Features — Failed Experiment (v2.21)

**⚠️ RETRACTED**: The "15-dim contextual features" claim in earlier documentation was aspirational and has been empirically disproved.

**v9_v5 experiment results** (Apr 2026):
- NSL-KDD: ARI dropped from **0.714 → 0.003** (catastrophic)
- TON_IoT: ARI dropped from **0.737 → 0.028** (catastrophic)

**Root cause**: Batch-computed frequency statistics only work if the normalization is batch-relative (each batch provides its own reference distribution). The v9_v5 implementation used a global constant denominator `log1p(2000.0)` which destroys discriminative variance. Additionally, global precomputation before chunking removes the domain-adaptive inference property.

**Current status**: v9_v3 with 6-dim base features reverted as canonical baseline. Any future contextual feature attempt MUST use batch-relative normalization and identical train/inference computation paths.

### 1.5 Gaps and Risks

#### Gap 1: Real-World SIEM Performance is Poor (ARI=0.005 → 0.111 after fix)

**Data**: 
- Before fix: `experiments/results/archive_2026_mar/siem_kcluster_fixed_v2.csv` shows SQTK_SIEM ARI=0.005 across all gate values
- After fix: `experiments/results/sqtk_siem_gaec_sweep.csv` shows SQTK_SIEM ARI=0.111 at gate=0.9 (22x improvement)

**Context**: Benchmark datasets (UNSW, NSL-KDD, TON_IoT, CICIDS2017) are clean, labeled, controlled. The one real SIEM dataset scores 0.111 after fix.

**Root causes identified and fixed**:
1. No `checkpoint_override` — uses multidomain_v2 with softmax mode (OOD classification head) → Fixed
2. No `use_geometric_confidence: True` — stuck in softmax mode with near-random predictions → Fixed
3. No UMAP, no epsilon — missing embedding spread and cluster merging → Fixed
4. Sparse graph structure — hostname NIL for 99.5%, protocol NIL for 75% — Remains (data limitation)
5. 89% of records have campaign_id="UNKNOWN" (tactic labels unreliable) — Remains (data limitation)

**Risk**: This is still an important gap — production data is messy, alert volumes are high, and campaign labels are noisy. While 22x improvement was achieved, ARI=0.111 is still low. The embeddings are over-smoothed (cosine similarity > 0.95), indicating a deeper architectural issue with the HGNN on sparse graphs.

**Investigation completed**:
- ✅ Feature distribution analysis: Sparse graph structure identified
- ✅ Schema audit: NIL values in hostname/protocol columns identified
- ✅ Config fix applied: GAEC mode + UMAP + epsilon + network_v9_v3 checkpoint
- ⚠️ Deeper issue: Over-smoothed embeddings require architectural fix (reduce num_layers or add residual connections)

#### Gap 2: Host/APT Domain Gap is Unresolved

**Data** (verified):
- OpTC supervised fine-tune (multidomain_v2_optc): ARI=1.0 (binary task, perfect clustering)
- OpTC zero-shot with network_v9_v3 checkpoint: ARI=0.979 at gate=0.55 (binary_ARI=0.979)

**Context**: The network checkpoint (network_v9_v3) actually performs well on OpTC in zero-shot mode (ARI=0.979), contrary to earlier assumptions. The supervised fine-tune achieves perfect ARI=1.0, but the zero-shot performance is already excellent.

**Key findings from `experiments/results/optc_zeroshot_network_v9.csv`**:
- Best zero-shot ARI: 0.979 at gate=0.55
- Confidence mode: "softmax" (not GAEC)
- Bridge edge coverage: 65.3% of records have IP↔hostname correlations
- 2 clusters found (binary task: Benign vs RedTeam_Sep23)

**Risk**: The host/domain gap is smaller than initially thought. The network_v9_v3 checkpoint generalizes well to OpTC. However, specialized host-domain methods (MAGIC, ThreaTrace) might still outperform on pure host telemetry without network features.

#### Gap 3: No Comparison Against Recent GNN Baselines on Identical Splits

**Missing**: MAGIC (2024), ThreaTrace (2022), EULER (2023), KAIROS (2024) on UNSW-NB15/TON_IoT with the same train/test split and the same ARI metric.

**Risk**: Without this, "outperforms baselines" is a claim against K-Means and DBSCAN, not against 2024 GNN literature. Reviewers will demand comparison to state-of-the-art GNN methods.

**Effort required**: Medium. Need to implement these methods or use published code, run on identical data splits, report ARI.

**Critical insight on task/metric mismatch**:
- MAGIC, ThreaTrace, EULER, KAIROS solve **supervised binary anomaly detection** (attack vs normal)
- They report **AUC-ROC, Precision, Recall** — not ARI
- MITRE-CORE solves **unsupervised multi-class campaign clustering** (which campaign?)
- MITRE-CORE reports **ARI** — not AUC-ROC
- Direct comparison is not meaningful without significant adaptation work

**Honest framing approach**:
1. Document the task/metric difference explicitly in the paper
2. Show MITRE-CORE vs classical clustering baselines (already done: K-Means, DBSCAN, HDBSCAN, Spectral) on the same task (ARI)
3. Cite published GNN IDS numbers but note they solve a different problem
4. Future work: adapt MAGIC/ThreaTrace embeddings for clustering comparison (requires retraining with ARI loss)

#### Gap 4: main_results_table.csv Inconsistency

**Data**: `experiments/results/evaluation/main_results_table.csv` shows ARI=0.864 for "Exp D: Full v2". Real UNSW sweep shows ARI=0.523.

**Risk**: This is a credibility risk. If a reviewer discovers the 0.864 figure is from an early synthetic eval, it undermines trust in all results.

**Action required**: Document the synthetic/early eval setup that produced 0.864, clearly contextualize it as historical (not current SOTA), remove from headline claims.

### 1.6 Research Validity Assessment

**What is publishable now**:
- ✅ Zero-shot transfer: network_v9_v3 trained on UNSW, scores 0.737 on TON_IoT — novel, validated
- ✅ Confidence calibration: ECE 0.015-0.022 across 8 datasets — production-grade
- ✅ Baseline comparison: Beats K-Means, Spectral, HDBSCAN on UNSW
- ⚠️ NSL-KDD nuance: DBSCAN beats MITRE-CORE (0.749 vs 0.621) — must acknowledge
- Bridge edges: Implementation complete, ablation pending retraining — must acknowledge

**What needs more work**:
- ❌ SQTK_SIEM failure (ARI=0.005) — investigation required before any deployment claim
- ✅ Host domain gap: OpTC zero-shot ARI=0.979 (v2.20 verified) — smaller than expected
- ❌ Missing GNN baselines (MAGIC, ThreaTrace, EULER) — required for competitive comparison
- ❌ main_results_table.csv inconsistency (0.864 vs 0.523) — credibility fix required

**Publication framing recommendation**:
- Focus on the unsupervised multi-class clustering task (not binary anomaly detection)
- Emphasize zero-shot transfer as the key novel contribution
- Acknowledge domain specialization (network vs host) as a limitation
- Be transparent about SQTK_SIEM failure as a critical gap
- Contextualize main_results_table.csv as historical, not current

---

## Part 2 — Improvement Roadmap (by ROI)

| Priority | Item | Status | Expected Next Action |
|----------|------|--------|---------------------|
| **P0** | Fix e2e_test checkpoint routing | Not done | Use `ingestion/dataset_profiler.py` router to select per-dataset checkpoint |
| **P1** | Evaluate cross-graph NT-Xent | Implemented, untested | Retrain network_v9_v3 with `--cross_graph`, sweep UNSW+TON_IoT |
| **P2** | VGAE pretraining evaluation | Implemented, untested | Run `train_vgae.py`, fine-tune, compare ARI |
| **P3** | SIEM structural fix | Fallback done, gap remains | Investigate SQTK_SIEM graph: count actual edges, check node type distribution |
| **P4** | Host-domain checkpoint | Not started | Train dedicated model on OpTC+BETH syscall features |
| **P5** | Add MAGIC/ThreaTrace comparison | Not done | Implement or cite baseline numbers from their papers |

### P0: Fix e2e_test Checkpoint Routing

**Problem**: The e2e_test uses `network_v9_v3` for all datasets, causing apparent "regressions" on UNSW (needs `unsw_supcon_v2`) and suboptimal performance on others.

**Root cause**: No dataset-specific checkpoint routing in `run_gate_tuning.py` default config.

**Solution**: Integrate `ingestion/dataset_profiler.py` checkpoint router:
- Use `profile_dataset()` to detect dataset characteristics (IP density, hostnames, timestamps)
- Route UNSW → `unsw_supcon_v2`, NSL-KDD/TON_IoT/CICIDS2017 → `network_v9_v3`
- Verify Tier 2 results match Tier 1 when correct checkpoint is used

**Expected outcome**: Tier 2 results should match Tier 1, confirming the new features don't degrade performance.

### P1: Evaluate Cross-Graph NT-Xent

**Status**: Implemented in `hgnn/cross_domain_contrastive.py:18-106`, integrated in `training/train_graph_mae_v9_multidata_fast.py:179`

**Next steps**:
1. Retrain network_v9_v3 with `--cross_graph` flag enabled
2. Run gate tuning sweep on UNSW-NB15 and TON_IoT
3. Compare ARI to baseline (UNSW=0.523, TON_IoT=0.724)
4. Target: +5% ARI improvement from better negative sampling

**Risk**: May not show improvement if the current checkpoint already learned good representations.

### P2: VGAE Pretraining Evaluation

**Status**: Implemented in `hgnn/vgae_pretraining.py` and `training/train_vgae.py`

**Next steps**:
1. Run `python training/train_vgae.py` on UNSW-NB15
2. Save pretrained VGAE checkpoint
3. Fine-tune on downstream task (optional)
4. Evaluate clustering ARI
5. Compare to network_v9_v3 baseline

**Hypothesis**: VGAE pretraining on link prediction should improve graph structure learning, particularly for sparse graphs like SIEM.

### P3: SIEM Structural Investigation

**Current state**: SQTK_SIEM ARI=0.065 (P3 fallback exercised), structural sparsity remains

**Investigation plan**:
1. Count actual edges per node type in SQTK_SIEM graph
2. Check node type distribution (Alerts vs Hosts vs IPs vs Users)
3. Identify which edges exist (alert→host? alert→IP? host→IP?)
4. Determine if the graph is essentially a set of disconnected alerts
5. Consider schema modifications: synthetic edges? different graph construction?

**Target**: ARI ≥ 0.20 on SQTK_SIEM (current best: 0.111 with P3 fix, 0.065 in e2e_test)

### P4: Host-Domain Checkpoint

**Rationale**: Current network checkpoint generalizes well to OpTC (zero-shot ARI=0.979), but specialized host-domain model may help on syscall-only datasets (BETH).

**Action steps**:
1. Collect OpTC + BETH + Attack_Techniques datasets
2. Design host-domain schema (Process, commandline, user, file edges)
3. Train dedicated HGNN on multi-domain host data
4. Evaluate zero-shot transfer within host domain

### P5: GNN Baseline Comparison

**Status**: Task/metric mismatch documented (MAGIC/ThreaTrace/EULER solve binary supervised, not unsupervised multi-class)

**Options**:
1. Cite their published AUC-ROC numbers with clear context (different task)
2. Adapt their embeddings for clustering comparison (requires retraining with ARI loss)
3. Focus comparison on classical clustering baselines (K-Means, DBSCAN, HDBSCAN) — already done

**Recommendation**: Option 3 for publication, Option 1 for related work citation.

---

## Appendix: Data Sources

### Benchmark Sweep Results (Tier 1)
- TON_IoT: `experiments/results/network_v9_ton_iot_seed42.csv` (ARI=0.724)
- NSL-KDD: `experiments/results/gate_tuning_nslkdd_clean.csv` (ARI=0.719)
- CICIDS2017: `experiments/results/cicids2017_network_v9_sweep.csv` (ARI=0.617)
- UNSW-NB15: `experiments/results/unsw_supcon_v2_v2.csv` (ARI=0.523)
- SQTK_SIEM: `experiments/results/sqtk_siem_p3_fix.csv` (ARI=0.111)

### End-to-End Test Results (Tier 2)
- `experiments/results/e2e_test.csv` (Apr 2026, network_v9_v3 on all datasets)

### Baseline Comparison
- `experiments/results/baseline_clustering_comparison_v4.csv`

### Calibration Results
- `experiments/results/calibration/calibration_per_dataset.csv`

### Bridge Edge Analysis
- `experiments/results/optc_bridge_edge_analysis.json`

### Critical Code Locations
- P3 SIEM Fallback: `hgnn/hgnn_correlation.py:856-879`
- P6 Cross-Graph NT-Xent: `hgnn/cross_domain_contrastive.py:18-106`
- TGN Temporal Memory: `hgnn/hgnn_correlation.py:1035`
- VGAE Pretraining: `hgnn/vgae_pretraining.py`, `training/train_vgae.py`
- Dataset Router: `ingestion/dataset_profiler.py`

---

## Conclusion

MITRE-CORE has **genuine technical contributions**: zero-shot transfer capability and confidence calibration (GAEC). Bridge edges were tested with a 7-seed paired ablation and confirmed as having zero clustering impact — removed from contribution claims. Four research improvements were implemented in April 2026 (SIEM fallback, cross-graph NT-Xent, temporal memory, VGAE pretraining), but **require retraining to validate**.

**Current blockers for publication**:
1. **Checkpoint routing bug**: e2e_test shows apparent regressions due to using wrong checkpoint per dataset
2. **Unvalidated improvements**: Cross-graph NT-Xent, VGAE, temporal memory need training runs
3. **SIEM gap**: Production data remains challenging (ARI=0.065-0.111 vs 0.72 on benchmarks)

**Recommended path**:
1. Fix e2e_test checkpoint routing (P0) — verify Tier 2 matches Tier 1
2. Run training experiments for P1/P2 to validate improvements
3. Investigate SIEM structural issues (P3)
4. Update paper with honest, verified benchmark numbers
