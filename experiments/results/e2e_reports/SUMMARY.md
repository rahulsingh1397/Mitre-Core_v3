# Executive Summary: MITRE-CORE Review Claims Verification (DEEP-DIVE)
**Date:** 2026-03-05 | **Method:** Full training pipeline audit + figure generation audit

## Revised Accuracy Scores

| # | Report | Original | After Deep-Dive | Change |
|---|--------|----------|-----------------|--------|
| 1 | Architecture (HeteroGAT, single layer, 6-dim, 128 hidden) | 100% | 100% | — |
| 2 | Dataset Support & Evaluation Metrics | 90% | 90% | — |
| 3 | Self-Supervised / Zero-Shot Learning | 55% | **35%** | ↓20 |
| 4 | Explainability Module | 95% | 95% | — |
| 5 | Design Decisions & Ablation Studies | 55% | **19%** | ↓36 |

**Revised Weighted Average: 68% (down from 79%)**

---

## Deep-Dive Findings

### SELF-SUPERVISED / ZERO-SHOT (35%)

**`NTXentLoss` is dead code.** The class exists in `contrastive_loss.py` but is never used in any canonical training run. The only script that imports it (`train_hybrid_v10.py`) is explicitly marked as failed by the authors: "Caused UNSW-NB15 regression (<0.02 ARI). Not in canonical pipeline."

**What actually trained the checkpoints:**
- `multidomain_v2`: SimCLR (`ContrastiveAlertLearner`) + supervised `CrossEntropyLoss` on campaign labels
- `network_v9_v3`: Custom `topological_ntxent_loss()` (67%) + SimCLR-style `CrossDomainContrastiveLoss` (33%)

**"Topological" = co-occurrence edges.** Alerts sharing IPs/hostnames get edges. No persistent homology, Betti numbers, or simplicial complexes. Standard GNN adjacency.

**Zero-shot works on only 2/5 datasets.** NSL-KDD (0.743) and UNSW-NB15 (0.538) work. TON_IoT (0.082), OpTC (0.048), SQTK_SIEM (0.184) fail.

### DESIGN DECISIONS & ABLATIONS (19%)

**Systematic fabrication discovered.** The evidence chain:

1. **15 ablation CSV files** — all contain identical placeholder values (ARI=0.782-0.786 for A/B/C, 0.862-0.866 for D). Every configuration produces the same result.

2. **`aggregate_results.py`** — reads placeholder CSVs, computes mean/std, creating a veneer of statistical rigor.

3. **`generate_figures.py`** — **THE SMOKING GUN.** All 10 figures use hardcoded values or mathematical formulas:
   - `aris = [0.777, 0.784, 0.814, 0.844, 0.864]` (hardcoded)
   - `acc_pre = confidences - 0.15 * np.sin(...)` (formula, not data)
   - `lat_uf = sizes**2 / 100` (formula, not measured)
   - 4×4 security matrix (hardcoded)

4. **Hardcoded values exceed all real results.** Claimed ablation ARI=0.864 vs. best real ARI=0.845 (TON_IoT prototype).

**"Zero-shot beats per-dataset training" is false on 3/5 datasets.** TON_IoT prototype ARI=0.845 vs zero-shot 0.082.

---

## Key Files Referenced

| File | Role |
|------|------|
| `scripts/analysis/generate_figures.py` | All 10 figures from hardcoded values |
| `experiments/results/ablation_studies/*.csv` | 15 placeholder CSV files |
| `scripts/analysis/aggregate_results.py` | Reads placeholders, computes stats |
| `hgnn/hgnn_training.py` | Trained `multidomain_v2` (SimCLR + CE) |
| `training/train_graph_mae_v9_multidata_fast.py` | Trained `network_v9_v3` (topo + SimCLR) |
| `training/train_hybrid_v10.py` | Failed experimental script (only NT-Xent user) |
| `hgnn/contrastive_loss.py` | NTXentLoss — dead code |
| `hgnn/cross_domain_contrastive.py` | CrossDomainContrastiveLoss, CrossGraphNTXentLoss |

---

## Overall Assessment

The architecture and explainability claims are accurate. The dataset support is mostly accurate. However:

1. **The self-supervised/zero-shot narrative is misleading.** NT-Xent is dead code. The canonical checkpoint uses supervised labels. Zero-shot transfer is unreliable.

2. **The experimental validation is systematically fabricated.** All ablation CSVs are placeholders. All figures use hardcoded values. The aggregation script creates false statistical rigor.

The review's claims about what the system does architecturally are ~90% accurate. The review's claims about how well it works and how that was validated are ~20% accurate.
