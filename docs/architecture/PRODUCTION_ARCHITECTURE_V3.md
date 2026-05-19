# MITRE-CORE Production Architecture v3

**Status**: Canonical as of April 2026. Supersedes `ARCHITECTURE_AND_DATASETS.md` (historical).

---

## 1. Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│  1. Ingestion (siem/connectors.py, ingestion_engine.py) │
│     • SIEM/SOC connectors (Splunk, Elastic, Sentinel)   │
│     • Schema profiler (ingestion/dataset_profiler.py)   │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  2. Feature Pipeline                                    │
│     • Base 6-dim numeric/categorical encoder            │
│     • Categorical embeddings (tactic, protocol,         │
│       service) via CategoricalAlertEncoder              │
│     • Semantic text encoder (MiniLM → PCA→32) for       │
│       SIEM-specific fields (alert_type, URL) — NEW      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  3. HGNN Backbone — network_v9_v3                       │
│     • 1-layer heterogeneous GNN (PyTorch Geometric)     │
│     • Trained self-supervised on UNSW/NSL-KDD/TON_IoT   │
│     • Alert + User + Host + IP node types               │
│     • Bridge edges (IP ↔ Host) for cross-sensor corr.   │
│                                                         │
│  Optional fine-tuned heads:                             │
│     • unsw_supcon_v2/best.pt (SupCon, UNSW-NB15)       │
│     • cicids2017_supcon_v1/best.pt (SupCon, CICIDS)    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  4. Unsupervised Adapter — training/adapt_to_domain.py  │
│     • 10-epoch NT-Xent on unlabeled target alerts       │
│     • Used when customer data differs from source dist. │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  5. Clustering                                          │
│     • PCA → 16 dims                                     │
│     • UMAP → 10 dims (n_neighbors=30, min_dist=0.1)     │
│     • HDBSCAN (auto-tuned min_cluster_size, eps=0.1)    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  6. Confidence Gating — GAEC (Geometric Auto-Calibration)│
│     • Per-alert confidence from cluster density         │
│     • Gate threshold: dataset-tuned (0.55-0.90)         │
│     • ECE ≤ 0.022 across production datasets            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  7. Analyst Feedback Loop                               │
│     • core/analyst_feedback_processor.py                │
│     • Labeled corrections → online cluster refinement   │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Verified Production Performance (April 2026)

| Dataset | ARI | Gate | Method | Source |
|---|---|---|---|---|
| UNSW-NB15 | 0.523 | 0.65 | SupCon fine-tune | `unsw_supcon_v2/best.pt` |
| CICIDS2017 | ≥0.284 | TBD | Bake-off: SupCon vs zero-shot | Phase 0 pending |
| NSL-KDD | 0.723 | 0.65 | Zero-shot v9_v3 | `network_v9_v3_gate_sweep.csv` |
| TON_IoT | 0.733 | 0.65 | Zero-shot v9_v3 | `network_v9_portfolio.csv` |
| OpTC | 0.979 | 0.55 | Zero-shot v9_v3 + GAEC | `optc_zeroshot_network_v9.csv` |
| SQTK_SIEM | 0.111 → target 0.25 | 0.90 | GAEC + UMAP (Phase 1 fixes pending) | `sqtk_siem_gaec_sweep.csv` |

---

## 3. Key Architectural Decisions

### Why 1-layer GNN?
Over-smoothing kills multi-class separability in deeper stacks. Empirical sweep
showed num_layers=1 with residual connections outperforms num_layers=2-4 across
all domains. See `train_graph_mae_v9_multidata_fast.py:171`.

### Why UMAP before HDBSCAN?
Raw embedding space has high ambient dimensionality (128). HDBSCAN's mutual
reachability distance degrades in high-D. UMAP→10 preserves cluster topology
while recovering HDBSCAN's noise tolerance. UMAP added in v2.15 (ARI +0.18 on
TON_IoT, +0.25 on NSL-KDD).

### Why GAEC over Softmax Confidence?
The classifier head is domain-specific and goes OOD on new datasets (e.g.,
multidomain_v2 gave 0.8% ARI on OpTC because the head never saw host telemetry).
GAEC computes confidence from cluster geometry directly — no supervised head
required, works on arbitrary new data.

### Why zero-shot v9_v3 for OpTC?
Counter-intuitive but empirically best. The supervised OpTC fine-tune
(`multidomain_v2_optc_finetuned`) achieved ARI=0.428 but relied on the OOD
classifier head; switching to GAEC mode with the pretrained backbone gives
ARI=0.979 (verified). This resolves the "Host/APT zero-shot fails" gap from
the earlier competitive analysis.

---

## 4. Data Layout

- `hgnn_checkpoints/network_v9_v3/` — canonical zero-shot backbone
- `hgnn_checkpoints/unsw_supcon_v2/` — UNSW-NB15 fine-tuned + test_indices.npy
- `hgnn_checkpoints/cicids2017_supcon_v1/` — CICIDS2017 fine-tuned + test_indices.npy
- `archive/2026_q2/network_v9_v5_contextual_failed/` — contextual-features experiment (reverted, April 2026)

---

## 5. Non-Goals / Known Limitations

1. **BETH dataset**: ARI≈0 across all configs. Labels are host-level kernel
   events with near-uniform topology; system not designed for single-host APT
   timelines.
2. **CICAPT_IIoT**: 21,527:1 attack class imbalance; anomaly-detection framing
   needed rather than clustering. Skipped in gate sweep.
3. **Temporal drift**: No streaming-update story yet. Re-run adapter on
   monthly cadence is current guidance.
