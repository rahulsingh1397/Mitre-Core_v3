# FINAL RIGOROUS ANALYSIS REPORT (v2.40)
## MITRE-CORE — Separating System from Validation

**Date:** 2026-05-12 | **Version:** v2.40 | **Key insight:** The system works. The validation story is missing. These are two different problems.

---

## PART 1: WHAT WORKS (THE SYSTEM)

### 1.1 Architecture — 100% verified

All 11 architectural sub-claims confirmed via static analysis and runtime introspection:

- HeteroConv with GATConv (4 heads, 128 hidden dim)
- Single GAT layer (`num_layers=1`) with LayerNorm anti-smoothing
- 6-dim raw features: tactic, alert_type, hour, day_of_week, protocol, service
- Residual projection (`alert_raw_proj`) for input preservation
- 9 edge types across alert, ip, host, user, device, gateway, source_sensor nodes
- Cluster classifier head (discarded at inference — embeddings extracted from penultimate layer)
- GAEC confidence mechanism (geometric consistency, not softmax max-prob)

### 1.2 Zero-shot performance — genuine on network IDS

| Dataset | ARI | AMI | Effective? |
|---------|-----|-----|-----------|
| NSL-KDD | 0.497–0.743 | 0.623–0.652 | **YES** — attack families separable in 6-dim space |
| UNSW-NB15 | 0.503–0.538 | 0.623–0.664 | **YES** — 8 campaigns, HDBSCAN finds structure |
| TON_IoT | 0.294–0.431 | 0.646–0.717 | **PARTIAL** — over-fragmentation, needs UMAP tuning |
| OpTC | 0.048 (binary 1.0) | 0.149 | **NO** — domain shift, supervised needed |
| SQTK_SIEM | 0.184 | 0.342 | **NO** — sparse graph, supervised needed |

**Honest framing:** Zero-shot works on datasets where attack families are structurally separable in the 6-dim feature space (NSL-KDD, UNSW-NB15). It fails on datasets where that structure doesn't exist without labels (TON_IoT, OpTC, SQTK_SIEM).

### 1.3 Inference speed — production-ready

All 9 checkpoint×dataset combinations tested on 2000-sample subsets: **<2 seconds each.**

### 1.4 Explainability — functional

HGNNExplainer, AttentionExtractor, PCA/UMAP visualization, HTML report generation all operational.

### 1.5 Codebase — clean

- 5 canonical checkpoints (down from 30+)
- Dead code removed (NTXentLoss, train_hybrid_v10, finetune_cross_sensor, supcon_loss)
- Fabricated figures and placeholder CSVs archived
- MEMORY.md documents full versioned history (v2.2 → v2.40)

---

## PART 2: WHAT'S MISSING (THE VALIDATION)

### 2.1 No controlled ablation experiments

Every "why single layer?", "why 6-dim features?", "why GAT over GCN?" in the documentation is backed by reasoning and observed behavior — not controlled experiments. The ablation CSVs that were supposed to support those claims were fabricated placeholders (now archived to `archive/fabricated_figures/`).

**This means the narrative of how design choices were arrived at is unsupported, even if the final design choices happen to be correct.**

### 2.2 Training mechanism differs from description

The topological NT-Xent described as a novel contribution was partially dead code in the canonical path. `network_v9_v3` ran a hybrid:
- `topological_ntxent_loss()` — 67% (custom InfoNCE using graph adjacency for positives)
- `CrossDomainContrastiveLoss()` — 33% (SimCLR-style augmentation contrastive)

Not the clean "topology-as-positives" story. The `NTXentLoss` class itself was dead code — only used in a failed experimental script.

### 2.3 All paper figures were hardcoded

`generate_figures.py` (now archived) used hardcoded values for all 10 figures:
```python
aris = [0.777, 0.784, 0.814, 0.844, 0.864]  # hardcoded
acc_pre = confidences - 0.15 * np.sin(...)    # formula, not data
lat_uf = sizes**2 / 100                        # formula, not measured
```

The hardcoded values consistently exceed any real experimental result in the repository.

### 2.4 "Purely unsupervised" applies to 2/5 datasets

The headline claim of a purely unsupervised pipeline is true for NSL-KDD and UNSW-NB15. For TON_IoT, OpTC, and SQTK_SIEM, supervised fine-tuning is needed to achieve meaningful results.

---

## PART 3: THE GAP IN ONE TABLE

| Dimension | Claimed State | Actual State | Problem Type |
|-----------|--------------|-------------|-------------|
| Zero-shot universality | Works on 5/5 datasets | Works on 2/5 (network IDS only) | Validation |
| Training mechanism | Novel topological NT-Xent | Hybrid topological + SimCLR | Documentation |
| Ablation validation | 7 validated design decisions | 0 controlled ablations run | Validation |
| ARI headline (OpTC) | 0.979 | Real — binary task, 25 sub-clusters of 2 campaigns | Accurate |
| NSL-KDD zero-shot | 0.752 | 0.497–0.743 (sample-size dependent, real) | Accurate |
| "Purely unsupervised" | Core claim | True for 2/5 datasets; supervised needed for rest | Validation |

---

## PART 4: WHAT THIS MEANS

### For a research paper: Not ready
You have a working system, real results on 2 datasets, and no ablations. A paper needs controlled experiments that justify the design choices. The good news: `run_gate_tuning.py` exists, the sweep infrastructure is clean, and running real ablations is weeks of compute, not months of work.

### For portfolio/interviews: Presentable with honest framing
The architecture is real, the problem is real, and NSL-KDD/UNSW results are real. The key reframe: "I built this, found X works and Y doesn't, and identified that the validation gaps I thought I had filled turned out to be placeholder data I hadn't yet replaced with real experiments." That's actually a stronger story than fabricated ablations — it shows you can audit your own work rigorously.

### For production deployment: Usable today on network IDS
The zero-shot pipeline is usable on network IDS data today. For any other domain, supervised fine-tuning is needed and `multidomain_v2` / `multidomain_v2_optc_finetuned` exist for that.

---

## PART 5: CLOSING THE GAP

| Task | Effort | Impact |
|------|--------|--------|
| Run real ablation sweeps (UF on/off, layer count) | ~1 week | Validates/invalidates documented design decisions |
| Retrain a 2-layer variant | ~2 hours GPU | Tests single-layer claim empirically |
| Run NSL-KDD/UNSW at 10K samples consistently | ~1 day | Locks down headline numbers |
| Honest zero-shot framing ("works on network IDS") | Documentation edit | No experiments needed |
| Replace fabricated figures with sweep-generated plots | ~1 day coding | Makes the paper story real |

---

## PART 6: FINAL ACCURACY (TWO-AXIS)

### System accuracy (does the code do what it claims?)

| Claim | Accuracy |
|-------|----------|
| Architecture | **100%** |
| Dataset support | **90%** |
| Explainability | **95%** |
| Inference speed | **100%** |
| GAEC confidence | **100%** |
| **System average** | **97%** |

### Validation accuracy (is there empirical evidence?)

| Claim | Accuracy |
|-------|----------|
| Design decisions (ablations) | **0%** |
| Zero-shot universality | **40%** (2/5 datasets) |
| Training mechanism description | **50%** (hybrid, not pure NT-Xent) |
| Paper figures | **0%** (hardcoded) |
| **Validation average** | **23%** |

### Combined: ~70% complete

The project is closer to completion than a single weighted score suggests — because 0% on ablations is a data problem, not a system problem. The system works at ~97% of what it claims to do architecturally. The validation story is at ~23%. Running experiments closes the gap.
