# MITRE-CORE — Claude Code Session Context

## What this project is (honest version)

A heterogeneous GNN trained with a topological contrastive objective that learns a 128-dim embedding space where network-layer attack families have geometric separation. On structurally clean datasets (NSL-KDD, UNSW-NB15), it clusters alerts into campaigns without labels at ARI ~0.5–0.74. On IoT and host telemetry, supervised fine-tuning is needed. The architecture is novel in its application to multi-sensor SOC alert correlation; the design choices are reasonable but empirically unvalidated. The explainability infrastructure and GAEC confidence mechanism are functional.

**This is a legitimate, interesting 70%-complete research project — not a failed one.**

## Two distinct problems

### 1. The system works (real)
- Ingests raw SIEM/IDS alerts and groups them into campaign clusters
- Genuine zero-shot performance on network IDS data (NSL-KDD ~0.74, UNSW ~0.54)
- Working confidence mechanism (GAEC) — not just softmax max-prob
- Runs in under 2 seconds on 2,000-alert batches
- Architecturally clean — 5 checkpoints, no dead code, documented edge types

### 2. The validation story is missing (data problem, not system problem)
- No controlled ablation experiments were run
- Design decisions (single layer, 6-dim features, GAT over GCN) are backed by reasoning and observed behavior, not experiments
- The ablation CSVs were fabricated placeholders — now archived
- The paper figures were hardcoded — now archived
- Closing this gap requires running experiments, not rewriting code

## Current state (v2.43, May 17, 2026 — Multi-Seed Validated)
Dead code removed, checkpoints cleaned, fabricated figures archived, documentation reframed.
5 canonical checkpoints retained. §3 headline numbers validated over 3 seeds (42, 43, 44).
See MEMORY.md v2.39–v2.40 for cleanup log and MITRE-design.md v2.43 for current verified results.

## Canonical checkpoints

| Checkpoint | Best For | ARI | Mode |
|-----------|----------|-----|------|
| `network_v9_v3/network_it_best.pt` | NSL-KDD, UNSW, TON_IoT, OpTC, CICIDS2017 | 0.739 (NSL-KDD) | Zero-shot GAEC |
| `siem_supcon_v4/best.pt` | SQTK_SIEM | 0.184 | GAEC |
| `unsw_supcon_v7/best.pt` | UNSW-NB15 semi-supervised | 0.538 | SupCon + Spectral |
| `multidomain_v2/best_supervised.pt` | Historical reference | 0.665 (UNSW) | Supervised softmax |
| `multidomain_v2_optc_finetuned/best_supervised.pt` | OpTC binary | 0.897 | Supervised softmax |

## Verified results (post-cleanup E2E sanity test, 2,000 samples, seed=42)

> **Note**: These are 2K-sample sanity checks, not the headline numbers. §3 headline table uses 10K stratified samples (or full corpus) over 3 seeds. See MITRE-design.md v2.43 for authoritative results.

| Dataset | network_v9_v3 | unsw_supcon_v7 | siem_supcon_v4 |
|---------|--------------|----------------|----------------|
| UNSW-NB15 | ARI=0.503, AMI=0.623 | ARI=0.534, AMI=0.645 | ARI=0.527, AMI=0.632 |
| NSL-KDD | ARI=0.497, AMI=0.623 | ARI=0.478, AMI=0.590 | ARI=0.443, AMI=0.571 |
| TON_IoT | ARI=0.294, AMI=0.646 | ARI=0.300, AMI=0.653 | ARI=0.293, AMI=0.643 |

## Claimed vs actual

| Dimension | Claimed | Actual |
|-----------|---------|--------|
| Zero-shot universality | Works on 5/5 datasets | Works on 2/6 (network IDS only); usable on 6/6 with per-dataset tuning |
| Training mechanism | Novel topological NT-Xent | Hybrid topological + SimCLR |
| Ablation validation | 7 validated design decisions | 0 controlled ablations run |
| NSL-KDD zero-shot | 0.752 | 0.739 (§3 verified, 10K samples, 3 seeds) |
| "Purely unsupervised" | Core claim | True for 2/6 datasets; supervised needed for rest |

## What closing the gap looks like

| Task | Effort | Impact |
|------|--------|--------|
| Run real ablation sweeps (UF on/off, layer count) | ~1 week | Validates/invalidates documented design decisions |
| Retrain a 2-layer variant | ~2 hours GPU | Tests single-layer claim empirically |
| Run NSL-KDD/UNSW at 10K samples consistently | ~1 day | Locks down headline numbers |
| Honest zero-shot framing ("works on network IDS") | Documentation edit | No experiments needed |
| Replace fabricated figures with sweep-generated plots | ~1 day coding | Makes the paper story real |

## Key files

- `training/train_graph_mae_v9_multidata_fast.py` — SSL training (topological + SimCLR hybrid)
- `training/finetune_supcon.py` — SupCon fine-tuning (has its own inline SupConLoss)
- `hgnn/hgnn_correlation.py` — main engine (HGNNCorrelationEngine + MITREHeteroGNN)
- `experiments/run_gate_tuning.py` — sweep runner
- `MEMORY.md` — full versioned experiment history (v2.2 → v2.40)

## Confirmed findings across all versions

- `use_uf_refinement=False` is correct default (UF is net-harmful)
- GAEC mode > softmax mode for clustering
- Single GAT layer avoids over-smoothing (observed, not experimentally validated)
- 6-dim base features are domain-agnostic and generalize better than 15-dim contextual
- Bridge edges and entity collapse: both zero effect — closed research directions
- CS fine-tuning: ineffective — closed research direction
- HDBSCAN seeding fixed — full reproducibility with seed=42
