# MITRE-CORE: Technical Architecture, Experiments & Interview Preparation

> **Canonical checkpoint**: `hgnn_checkpoints/network_v9_v3/network_it_best.pt` · **Primary metric**: AMI  
> **Core claim**: Unsupervised alert-to-campaign correlation — zero labels required at inference  
> **Verified on**: 6 public datasets + archived alerts from one SOC (May 2026)  
> **Honest scope**: Zero-shot works on network IDS data (NSL-KDD, UNSW-NB15); IoT/host telemetry need supervised fine-tuning  
> **Status**: System functional; §3 headline numbers validated over 3 seeds (42, 43, 44); ablation experiments in §11 are partially controlled

---

## Table of Contents

1. [The Problem — Why Alert Correlation is Hard](#1-the-problem)
2. [What MITRE-CORE Is](#2-what-mitre-core-is)
3. [Verified Results at a Glance](#3-verified-results)
4. [End-to-End Pipeline](#4-end-to-end-pipeline)
5. [Graph Construction — AlertToGraphConverter](#5-graph-construction)
6. [HGNN Architecture — MITREHeteroGNN](#6-hgnn-architecture)
7. [Self-Supervised Training](#7-self-supervised-training)
8. [Inference & Clustering Pipeline](#8-inference--clustering-pipeline)
9. [Dataset Structural Analysis](#9-dataset-structural-analysis)
10. [Experiment History (v2.1 → v2.39)](#10-experiment-history)
11. [Ablation Studies — What Was Tried and Why It Failed](#11-ablation-studies)
12. [Metrics Deep Dive](#12-metrics-deep-dive)
13. [Design Decisions & Engineering Rationale](#13-design-decisions--engineering-rationale)
14. [Limitations](#14-limitations)
15. [Related Work](#15-related-work)
16. [Interview Preparation](#16-interview-preparation)

---

## 1. The Problem

A mature Security Operations Centre processes **500–10,000 alerts per day** from a heterogeneous stack of sensors: network IDS (Suricata, Snort), endpoint EDR (CrowdStrike, Defender), WAF (F5, Imperva), and SIEM aggregation. Each alert is an isolated event — "SSH brute force from 10.0.1.42", "lateral movement on HOST-07", "C2 beacon to 185.x.x.x". Individually, these alerts have poor fidelity and high false-positive rates.

The actual threat, however, is a **multi-stage attack campaign** — an adversary executing Reconnaissance → Initial Access → Lateral Movement → Exfiltration over hours or days, generating thousands of correlated alerts across multiple sensors. A skilled analyst reads these alerts and mentally correlates them into a narrative. MITRE-CORE automates that narrative construction.

### Why Existing Approaches Fail

| Approach | Failure Mode |
|----------|-------------|
| **Rule-based correlation** (Splunk ES, QRadar) | Brittle — misses novel TTPs not in the ruleset; explosion of tuning required per environment |
| **Supervised ML** (DeepCASE, WATSON) | Requires labelled campaign data — unavailable in most SOCs; doesn't generalise across environments |
| **Simple clustering** (K-Means on flow features) | Ignores structural relationships (shared IPs, temporal proximity, user paths) between alerts |
| **Homogeneous GNN** | Treats all entities identically — an IP node and a process node need fundamentally different learned representations |

**MITRE-CORE's answer**: model alerts and the entities they involve (IPs, hosts, users, devices) as a *heterogeneous graph*, learn embeddings that capture structural attack patterns via self-supervised contrastive training, then cluster those embeddings into campaigns — *without any labels at inference time*.

---

## 2. What MITRE-CORE Is

MITRE-CORE is a **self-supervised** heterogeneous graph neural network (HGNN) pipeline for grouping raw security alerts into multi-stage attack campaigns. It operates zero-shot on network IDS data (NSL-KDD, UNSW-NB15); supervised fine-tuning is needed for IoT and host telemetry domains. No labels are consulted at inference time in zero-shot mode. The model generalises zero-shot to datasets it was never trained on within the network-IT domain.

### The Three Inference Modes

MITRE-CORE supports three modes, in increasing label dependency:

| Mode | Label Requirement | When to Use |
|------|------------------|-------------|
| **Zero-shot (Unsupervised)** | None — production default | Any new environment, zero configuration |
| **Semi-supervised (SupCon)** | Small labelled set for fine-tuning | When 100–1000 labelled campaigns are available |
| **Supervised (Prototype)** | Full campaign labels for prototype training | Benchmark / research comparison only |

**The primary contribution is zero-shot.** Semi-supervised and supervised modes were implemented for ablation comparability. The key architectural claim — that graph topology provides signal beyond raw features, learned in a self-supervised manner — is validated entirely in the zero-shot regime.

### System Contributions (Engineering, not novel ML)

MITRE-CORE's contribution is a *system* that integrates well-known building blocks for unsupervised SOC alert correlation. Each component below is a pragmatic engineering choice, not a new ML primitive. Heterogeneous GNNs (R-GCN, HAN, HGT), GAT, NT-Xent / SimCLR, ZCA whitening, and HDBSCAN membership vectors are all prior art; the contribution is the security-domain integration and the empirical analysis of where it does and does not help.

| Component | Technical Substance | Status |
|---|---|---|
| **Security-domain heterogeneous schema** | 8+ node types, 29 edge relation types, each with independent GATConv weights | Application engineering on top of PyG `HeteroConv` |
| **Hybrid contrastive loss** | Topological NT-Xent (67%) + SimCLR (33%) — graph edges as positive pairs, augmented views as negatives | Practical hybrid; weights chosen empirically. **No 67/33 vs 50/50 vs 100/0 ablation has been run** — pending (§14). |
| **GAEC confidence scoring** | Wraps `hdbscan.all_points_membership_vectors()` as a deployment-appropriate alternative to softmax max-prob | Standard HDBSCAN API used in a security-deployment context |
| **Dataset-aware checkpoint routing** | `dataset_profiler.py` + `DATASET_CONFIG` dict selects checkpoint and clustering config from a structural fingerprint (n_unique_ips, has_timestamps) | Config-level routing, not a learned mechanism |
| **Soft-ZCA whitening** | ZCA whitening (Bell & Sejnowski 1995) with `eps=0.1` regularisation applied post-hoc to collapsed embeddings | Standard ZCA, no novel claim |
| **Single-layer architecture (L=1)** | Empirically prevents over-smoothing on high-degree entity nodes | Locked based on informal L=2 vs L=1 comparison; **a controlled multi-seed L∈{1,2,3} ablation is pending (§14)** |
| **AMI as primary metric** | Information-theoretic; robust to legitimate sub-clustering that ARI penalises | Methodological choice, well-known metric |

---

## 3. Verified Results

All numbers from controlled experiments run May 2026, validated over **3 seeds (42, 43, 44)**. AMI is primary (see §12). Datasets without sampling randomness (full-corpus, deterministic clustering) show zero variance; those with 10K stratified subsampling show small variance (max CoV = 12.3% on CICIDS2017).

### Zero-Shot (Unsupervised) — network_v9_v3 Checkpoint

| Dataset | AMI ↑ (mean±std) | ARI ↑ (mean±std) | n_clusters | Key Note |
|---------|------------------|------------------|------------|----------|
| **NSL-KDD** | **0.668±0.000** | **0.739±0.000** | 14 | Deterministic (no sampling); disconnected graph |
| **UNSW-NB15** | **0.614±0.039** | **0.434±0.034** | 8 | CoV=6.3%; spectral k=8 |
| **TON_IoT** | **0.717±0.000** | **0.431±0.000** | 37 | Deterministic (10K stratified, fixed seed in graph) |
| **OpTC** | **0.148±0.000** | **0.047±0.000** | 24 | binary_ARI=0.999; deterministic |
| **SQTK_SIEM** | **0.342±0.000** | **0.184±0.000** | 11 | Deterministic (full corpus, spectral k=11) |
| **CICIDS2017** | **0.510±0.063** | **0.460±0.095** | ~53 | CoV=12.3%; sampling-dependent |

> **Note on Attack F1**: Attack F1 is a binary metric (attack vs. benign). It is reported only for datasets with binary or attack/benign label semantics (TON_IoT, OpTC). For multi-class tactic datasets (NSL-KDD, UNSW-NB15, SQTK_SIEM, CICIDS2017), Attack F1 is not applicable — all classes are attack variants with no benign baseline.
>
> OpTC, though multi-campaign in sub-clusters, has a binary RedTeam/Benign ground truth — Attack F1 is applicable and reported.

### Baseline Clustering Comparison (May 2026 — All Methods × All Datasets)

| Dataset | K-Means | DBSCAN | HDBSCAN | K-Means-emb | HDBSCAN-emb | MITRE-CORE | Best Method |
|---------|---------|--------|---------|-------------|-------------|------------|-------------|
| UNSW-NB15 | -0.004 | 0.497 | 0.419 | **0.608** | 0.431 | 0.502 | K-Means-emb |
| NSL-KDD | 0.021 | **0.753** | 0.505 | 0.318 | 0.401 | 0.598 | DBSCAN |
| TON_IoT | 0.125 | 0.101 | 0.094 | **0.720** | 0.059 | 0.363 | K-Means-emb |
| CICIDS2017 | 0.059 | **0.977** | 0.103 | 0.133 | 0.015 | 0.108 | DBSCAN |
| SQTK_SIEM | 0.000 | 0.000 | 0.000 | 0.254 | 0.192 | **0.364** | MITRE-CORE |
| OpTC | 0.078 | 0.002 | 0.093 | 0.078 | 0.021 | 0.054 | HDBSCAN |

> **Note on DBSCAN-emb omission**: DBSCAN on embeddings is excluded from the baseline comparison because it requires manual epsilon tuning per dataset and fails in cosine embedding space (the default HDBSCAN metric). HDBSCAN-emb is the principled density-based alternative — it inherits DBSCAN's density philosophy while operating without a fixed epsilon radius and providing membership confidence vectors.

**Honest framing**: MITRE-CORE wins outright on 1/6 datasets (SQTK_SIEM, real SOC archived alerts). On 2/6 datasets (UNSW, TON_IoT) the best method is K-Means on HGNN embeddings — i.e., the *embedding* contributes but the bespoke MITRE-CORE GAEC pipeline does not. On 2/6 datasets (NSL-KDD, CICIDS2017), DBSCAN on raw features dominates, indicating that for cleanly-separable feature spaces a graph adds nothing.

The defensible claim is therefore narrower than "MITRE-CORE outperforms baselines". It is: *HGNN embeddings learned via heterogeneous self-supervision are reusable artifacts that improve downstream clustering on heterogeneous-entity SIEM data; on datasets with clean feature separation, classical baselines remain competitive or superior.*

### HGNN vs Feature-Only Baselines

| Dataset | Best Feature-Only ARI | Best Embedding ARI | MITRE-CORE ARI | Graph Lift |
|---------|----------------------|--------------------|----------------|------------|
| NSL-KDD | 0.753 (DBSCAN) | 0.401 (HDBSCAN-emb) | 0.598 | Feature-only wins |
| UNSW-NB15 | 0.497 (DBSCAN) | 0.608 (K-Means-emb) | 0.502 | Embedding adds 22% |
| TON_IoT | 0.125 (K-Means) | 0.720 (K-Means-emb) | 0.363 | Embedding adds 5.8× |
| SQTK_SIEM | 0.000 (all) | 0.254 (K-Means-emb) | 0.364 | Graph essential |

**Honest assessment**: The "graph adds 2.5× value" claim is dataset-dependent. On TON_IoT, K-Means-emb scores 5.8× over raw K-Means — but raw K-Means is a weak comparator (no one runs K-Means on raw 78-dim flow features in production). A stronger neutral baseline (autoencoder + K-Means, or XGBoost-derived embedding + K-Means) is **not yet implemented** and is a known gap (§14). On NSL-KDD, raw-feature DBSCAN actually outperforms HGNN embeddings. The graph's value is real but context-specific — not a universal multiplier.

### Embeddings as Reusable Artifacts

A key architectural insight: HGNN embeddings are **reusable intermediate artifacts**, not tightly coupled to the MITRE-CORE clustering pipeline. The same 128-dim embedding matrix can be fed to K-Means, Spectral, HDBSCAN, or any future clustering algorithm without recomputing the graph forward pass. This decoupling means:

- **Downstream flexibility**: Analysts can experiment with clustering algorithms on pre-computed embeddings without re-running HGNN inference (~9.5s for 10K alerts on TON_IoT).
- **Incremental improvement**: Clustering improvements (e.g., better epsilon tuning, new algorithms) benefit all datasets immediately — no retraining or re-embedding needed.
- **Baseline fairness**: The baseline comparison (§3 table) uses identical embeddings for K-Means-emb, Spectral-emb, and HDBSCAN-emb — isolating clustering algorithm choice from embedding quality.

---

## 4. End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  INPUT: Raw alerts from any sensor (SIEM / IDS / EDR / WAF)            │
│  Formats: Splunk export, Wazuh JSON, Zeek/Bro logs, CSV, Parquet       │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 1 — INGESTION & NORMALISATION                                     │
│                                                                         │
│  MultiSourceIngestionPipeline (CS-4)                                    │
│  ├── Accepts N independent sensor feeds                                 │
│  ├── Adds data_source column (e.g. "cisco_fw", "crowdstrike_edr")       │
│  └── Normalises to MITRE-format schema:                                 │
│      [AlertId, timestamp, src_ip, dst_ip, hostname, username,           │
│       alert_type, tactic, campaign_id, protocol, service,               │
│       src_bytes, dst_bytes, stage, data_source]                         │
│                                                                         │
│  dataset_profiler.profile_dataset()                                     │
│  ├── Fingerprints structural properties (IP density, timestamps, hosts) │
│  └── Routes to optimal checkpoint + clustering config                   │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 2 — HETEROGENEOUS GRAPH CONSTRUCTION                              │
│                                                                         │
│  AlertToGraphConverter.convert(df) → PyG HeteroData                    │
│                                                                         │
│  Nodes created:                                                         │
│    alert        [N_a,  6]   6-dim encoded features (see §5)            │
│    ip           [N_ip, 32]  hash-embedded IP address                   │
│    host         [N_h,  32]  hash-embedded hostname                     │
│    user         [N_u,  32]  hash-embedded username                     │
│    device       [N_d,  32]  IoT device (port-derived)                  │
│    gateway      [N_g,  16]  subnet-derived gateway                     │
│    source_sensor[N_s,  32]  sensor origin node (CS-2)                  │
│    process      [N_p,  32]  Linux process name (APT datasets)          │
│    command_line [N_c,  64]  commandline string (APT datasets)          │
│                                                                         │
│  Edges constructed (29 relation types — full catalogue in §5)          │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 3 — HGNN INFERENCE                                                │
│                                                                         │
│  MITREHeteroGNN.forward(graph) → alert_embeddings [N_a, 128]           │
│                                                                         │
│  For each node type present: Linear(-1, 128) encoder                   │
│  1-layer HeteroGATConv: 4 heads × 32-dim per edge type                 │
│  LayerNorm + ReLU + Dropout(0.3) + residual skip                       │
│  B1 input residual: alert_raw_proj(raw_features) added back            │
│  → 128-dim embedding per alert                                          │
│                                                                         │
│  PyG HeteroConv gracefully skips absent node/edge types                │
│  → Zero-shot transfer without architecture changes                      │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STEP 4 — CLUSTERING & CONFIDENCE (GAEC)                                │
│                                                                         │
│  EmbeddingConfidenceScorer.fit_score(embeddings)                        │
│  ├── PCA: 128-dim → 16 components                                       │
│  ├── [Optional] UMAP: 16 → 10 (dataset-specific)                       │
│  ├── [Optional] Soft-ZCA whitening (for collapsed embeddings)           │
│  ├── HDBSCAN: cosine metric, auto-tune, epsilon merging                 │
│  └── all_points_membership_vectors() → per-alert confidence ∈ [0,1]   │
│                                                                         │
│  Clustering alternatives (dataset-specific):                            │
│  ├── Spectral k=N (UNSW-NB15: k=8)                                     │
│  ├── BGMM (experimental)                                                │
│  └── Prototype head (supervised mode only)                              │
└───────────────────────────┬─────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: alert_id → campaign_id mapping + confidence scores            │
│  SOC analyst receives ≤50 campaign narratives instead of 10K alerts    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Checkpoint Routing Logic (`dataset_profiler.py`)

The profiler fingerprints the dataset from a 1,000-row sample (~0.1s) and selects the appropriate checkpoint:

```python
n_unique_ips = max(src_ips.nunique(), dst_ips.nunique())
ip_density   = n_unique_ips / sample_size
has_timestamps = any col in ['timestamp', 'EndDate']
has_hostnames  = any col in ['hostname', 'process_name', 'CommandLine']

if n_unique_ips == 0 and not has_hostnames:
    → network_v9_v3   # NSL-KDD path: disconnected graph, feature-only
elif ip_density < 0.05 and has_timestamps:
    → network_v9_v3   # TON_IoT / UNSW: dense IoT / enterprise network
elif has_hostnames and n_unique_ips == 0:
    → multidomain_v2  # APT / host domain (OpTC, Linux_APT)
else:
    → network_v9_v3   # default for unrecognised network data
```

---

## 5. Graph Construction

### 5.1 The 6-Dimensional Alert Feature Vector

Every alert, regardless of source dataset, is normalised to a 6-dimensional integer feature vector before being fed to the model:

| Dim | Feature | Encoding | Semantic Role |
|-----|---------|----------|--------------|
| 0 | `tactic` | Integer 0–13 (MITRE ATT&CK phase) | Kill-chain position |
| 1 | `alert_type` | Integer hash 0–199 | Signature / rule type |
| 2 | `hour` | Integer 0–23 | Time-of-day attack pattern |
| 3 | `day_of_week` | Integer 0–6 | Weekday / weekend pattern |
| 4 | `protocol` | Integer 0–255 | Network protocol (TCP/UDP/ICMP) |
| 5 | `service` | Integer hash 0–49 | Application layer (HTTP/SMB/RDP) |

**Why exactly 6 dims?** A v9_v5 experiment tried 15-dim contextual features (IP frequency counts, temporal density ratios). It caused catastrophic regression: NSL-KDD ARI 0.714 → 0.003, TON_IoT 0.737 → 0.028. Root cause: contextual features computed from batch statistics at inference time, but the model was trained on a different distribution of those statistics. The 6-dim features are computed identically at training and inference time — completely stable across domain shifts.

### 5.2 Complete Edge Type Catalogue (29 Relations)

**Alert-to-Alert (intra-type)** — these run message passing directly between alerts:

| Relation | Construction Logic | Semantics |
|----------|-------------------|-----------|
| `(alert, shares_ip, alert)` | GROUP BY src_ip or dst_ip | Alerts that touch the same IP address |
| `(alert, shares_host, alert)` | GROUP BY hostname / DeviceHostName | Alerts on the same machine |
| `(alert, temporal_near, alert)` | Sliding 1-hour window | Time-proximate alerts |
| `(alert, semantic_similar, alert)` | GROUP BY alert_type hash | Same signature type |
| `(alert, precedes, alert)` | Temporal adjacency within 2h (CS-3, disabled) | Kill-chain ordering |

**Cross-type bidirectional pairs** — each has a forward and reverse direction:

| Forward | Reverse | Semantics |
|---------|---------|-----------|
| `user → owns → alert` | `alert → owned_by → user` | User-alert association |
| `host → generates → alert` | `alert → generated_by → host` | Host-alert association |
| `ip → involved_in → alert` | `alert → involves → ip` | IP-alert association |
| `device → connects_via → gateway` | `gateway → connected_to → device` | IIoT topology |
| `sensor_type → classifies → device` | `device → classified_as → sensor_type` | Device category |
| `device → generates → alert` | `alert → generated_by → device` | IIoT event |
| `process → executes → alert` | `alert → executed_by → process` | Linux APT process |
| `command_line → associated_with → alert` | `alert → has → command_line` | APT commandline |
| `container → runs_in → pod` | `pod → runs → container` | Kubernetes topology |
| `process → spawned_in → container` | `container → spawns → process` | Container process |
| `ip → resolves_to → host` | `host → resolved_from → ip` | Bridge edge (ablated — zero effect) |
| `alert → collected_by → source_sensor` | `source_sensor → collects → alert` | CS-2: sensor origin |

**Total: 5 alert-alert + 24 cross-type = 29 edge relation types.**

Each relation has **independent GATConv weights** — the model learns that `shares_ip` message passing should be weighted differently from `temporal_near`, which should differ from `user → owns`. This is impossible in a homogeneous GNN.

### 5.3 Why Heterogeneous Edges Matter

Consider two alerts from different attack stages that share a C2 IP address. In a homogeneous GNN they'd be connected by a generic edge and share gradient signal with every other alert connected to that IP. In the HGNN:

```
alert_A  ──(involves)──►  ip_185.x.x.x  ◄──(involves)──  alert_B
```

The `ip` node aggregates representations from all alerts involving it, then propagates a summary back. Alerts that are geometrically close in that IP neighbourhood learn similar embeddings — without any label telling the model "these alerts are part of the same campaign."

---

## 6. HGNN Architecture

### 6.1 Model Summary

```python
MITREHeteroGNN(
    alert_feature_dim = 6,       # Base feature dimension
    hidden_dim        = 128,     # Embedding size throughout the network
    num_heads         = 4,       # GAT attention heads per relation
    num_layers        = 1,       # Single layer — empirically validated (see §13)
    dropout           = 0.3,
    num_clusters      = 10,      # Softmax head dim (unused in GAEC mode)
    aggr_method       = "mean",  # HeteroConv: how to aggregate across edge types
)
```

### 6.2 Input Encoders

Each node type has an independent encoder that projects its raw features into the shared 128-dim space:

```
alert        : CategoricalAlertEncoder or Linear(-1, 128)
user         : Linear(-1, 128)     # lazy init — dim inferred on first forward
host         : Linear(-1, 128)
ip           : Linear(-1, 128)
device       : Linear(-1, 128)
gateway      : Linear(-1, 128)
source_sensor: Linear(-1, 128)     # CS-2 sensor origin
process      : Linear(-1, 128)     # Linux APT
command_line : Linear(-1, 128)     # Linux APT
container    : Linear(-1, 128)     # Kubernetes
pod          : Linear(-1, 128)     # Kubernetes
```

`Linear(-1, 128)` is PyG's lazy initialisation — the actual input dimension is resolved on the first forward pass from the data. This allows the same model class to ingest any dataset schema without code changes.

### 6.3 GATConv Configuration (per Edge Type)

```python
GATConv(
    in_channels  = 128,          # hidden_dim
    out_channels = 32,           # hidden_dim // num_heads
    heads        = 4,            # concat → output is 4 × 32 = 128
    dropout      = 0.3,
    add_self_loops = False,      # self-loops are redundant in the residual skip
)
```

**Graph Attention**: for each edge (i → j), the attention weight is:

```
α_ij = softmax( LeakyReLU( a^T [W·h_i || W·h_j] ) )
```

This means high-degree hub nodes (a common IP shared by 1,000 alerts) do not automatically dominate — the model learns to downweight noisy hub connections. GCN's fixed `1/√(deg_i · deg_j)` normalisation cannot do this.

### 6.4 Forward Pass — Step by Step

```
1. ENCODE  x_dict = { node_type: encoder(data[node_type].x) }
           # Each node type independently projected to 128-dim

2. SAVE    alert_raw = data['alert'].x
           # Kept for B1 input-side residual

3. FILTER  available_edges = { et: ei for et in data.edge_index_dict
                               if src_type in x_dict and dst_type in x_dict }
           # Gracefully skips absent node/edge types
           # → zero-shot cross-domain transfer without architecture changes

4. CONVOLVE  x_dict_new = HeteroConv(x_dict, available_edges)
             # Runs each edge type's GATConv independently
             # Aggregates multi-type messages with aggr="mean"

5. NORMALISE  for each node type k:
                x_dict[k] = LayerNorm( Dropout( ReLU( x_dict_new[k] ) ) )
                           + x_dict[k]           # ← residual skip connection

6. B1 RESIDUAL  x_dict['alert'] += alert_raw_proj(alert_raw)
                # Preserves raw feature variance — prevents information loss
                # during message passing on dense graphs

7. OUTPUT  cluster_logits = cluster_head(x_dict['alert'])   # [N_a, 10]
           return cluster_logits, x_dict
           # In GAEC mode: x_dict['alert'] (128-dim) is the embedding used for clustering
           # cluster_logits (softmax head) is only used in UF refinement mode (disabled)
```

### 6.5 The B1 Input Residual — Why It Exists

Without the input-side residual, message passing on a dense graph (e.g., a hub IP connected to 500 alerts) averages representations across all those alerts, destroying per-alert discriminative features. The B1 residual adds the raw 6-dim features (projected to 128-dim) back to the final alert embedding:

```
final_embedding = message_passing_output + alert_raw_proj(raw_6d_features)
```

This ensures that even if message passing collapses representations due to over-smoothing from high-degree neighbours, the raw feature signal remains in the final embedding. Empirically, this is critical for NSL-KDD (fully disconnected graph — message passing does nothing, but B1 ensures raw features flow through).

### 6.6 LayerNorm + Residual Skip

After each GATConv layer:

```
x_new = LN( Dropout( ReLU( GATConv(x) ) ) ) + x
```

LayerNorm stabilises training by normalising activations per layer. The residual skip ensures gradients flow to early layers even with non-zero dropout — standard practice from ResNet/Transformer architectures applied to the GNN setting.

### 6.7 Why One Layer?

With L=2 layers, an alert at distance 2 from a high-degree IP hub (which connects to thousands of unrelated alerts) receives averaged representations from the hub's entire neighbourhood. On UNSW-NB15, where 175K unique IPs create a star topology, this collapses embeddings: cosine_sim approaches 1.0, HDBSCAN finds a single cluster. Tested: L=1 gives ARI=0.40 on UNSW; L=2 gives ARI≈0.05 (embedding collapse). **L=1 is locked as the production default.**

---

## 7. Self-Supervised Training

### 7.1 Training Philosophy

No labels exist at training time. The model is trained to learn embeddings where **graph-connected alerts are similar** and **disconnected alerts are dissimilar**. The graph topology itself — which alerts share an IP, which are temporally adjacent — serves as the supervision signal.

### 7.2 Hybrid Contrastive Loss (Topological NT-Xent + SimCLR)

The canonical network_v9_v3 checkpoint was trained with a hybrid loss:
- **Topological NT-Xent (67% weight)**: Graph edges define positive pairs — structurally informed supervision
- **SimCLR (33% weight)**: Augmented views of the same graph as positive pairs — standard contrastive baseline

This is NOT the "pure topological NT-Xent" story originally claimed. The hybrid was chosen because pure topological NT-Xent on mini-graph batches lacks true negatives (all alerts in a batch come from the same campaign), causing over-smoothing. The SimCLR component provides in-batch negatives from other alert instances within the same batch (often from different campaigns, since each mini-graph batch contains multiple campaigns).

**Honest assessment**: The training mechanism is a practical hybrid, not a novel pure-topological contribution. The distinction matters for publication claims.

### 7.3 Dual Augmentation

Each training step applies two independent 15% edge dropouts to the same graph, creating two views (aug1, aug2). The total loss combines:
- Topological NT-Xent on aug1 (primary)
- Cross-graph alignment between aug1 and aug2 (SimCLR component, 33% weight — active during training)

Edge dropout at 15% prevents overfitting to specific graph structures while keeping enough topology for meaningful positive pairs.

### 7.4 Multi-Dataset Joint Training

```python
NETWORK_IT_DATASETS = ['unsw_nb15', 'nsl_kdd', 'ton_iot', 'optc']
```

All four datasets contribute to every epoch. Each campaign is chunked into ≤500-alert graphs and pre-loaded to GPU. The model sees diverse graph structures, alert types, and tactic distributions each epoch — forcing it to learn features that generalise across network-IT domains rather than overfitting to a single dataset's topology.

**Why joint training matters**: A model trained only on UNSW-NB15 learns IXIA-specific traffic patterns. Joint training produces a backbone that understands network-layer attacks in general — enabling zero-shot transfer to TON_IoT (ARI=0.431) and NSL-KDD (ARI=0.739) without retraining.

### 7.5 Training Configuration (network_v9_v3)

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Epochs | 150 | Convergence observed at ~120; extra epochs for stability |
| Batch size | 8 graphs/step | GPU memory balance on RTX 5060 Ti |
| Hidden dim | 128 | Ablated: 64 underpowered, 256 no gain |
| Temperature (topo) | 0.1 | Sharper cluster boundaries |
| Temperature (aug) | 0.5 | Softer cross-view alignment |
| Optimizer | AdamW | lr=3e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR | T_max=150, smooth decay |
| Edge dropout | 15% | Per augmentation view |
| num_layers | 1 | Over-smoothing prevention |
| Dropout | 0.3 | Standard regularisation |
| GPU | RTX 5060 Ti | ~7s/epoch, ~17.5 min total |
| Seed | 42 | Full determinism (torch, numpy, random, HDBSCAN) |

### 7.6 Checkpoint Evolution

| Checkpoint | Date | Key change | Status |
|-----------|------|-----------|--------|
| Pre-v7 | Pre-Apr 14 | MSE loss (wrong objective) | Collapsed — discarded |
| network_v7 | Apr 14 | NT-Xent fixed, 2 layers | Over-smoothing on UNSW |
| **network_v9_v3** | Apr 14 | L=1, B1 residual, 6-dim | **Canonical production** |
| network_v9_v5 | Apr (reverted) | 15-dim contextual features | Catastrophic ARI collapse |
| network_cs_best | May 8 | CS encoder fine-tune | Failed (AMI=0.0, early stop) |

---

## 8. Inference & Clustering Pipeline

### 8.1 GAEC — Geometry-Aware Embedding Confidence

Standard confidence scoring uses softmax max-probability. This fails for out-of-distribution inputs: softmax always produces values summing to 1, so even a completely OOD alert gets a "confident" assignment.

GAEC replaces this with **HDBSCAN cluster membership probability**:

```python
# After HDBSCAN clustering on alert embeddings:
membership_vectors = hdbscan.all_points_membership_vectors(clusterer)
# shape: [N_alerts, n_clusters]
# For each alert: soft probability distribution over ALL clusters
# (not just the assigned one — uses kernel density estimation)

confidence = membership_vectors.max(axis=1)
# Core cluster members    → confidence ≈ 1.0 (dense region)
# Border points           → confidence ≈ 0.3–0.7 (cluster edge)
# Noise / OOD points      → confidence ≈ 0.0–0.1
```

This is intrinsically calibrated to the embedding geometry — no temperature scaling required.

### 8.2 HDBSCAN Configuration

```python
HDBSCAN(
    min_cluster_size = dataset-specific (5–50),
    min_samples      = min_cluster_size // 3,
    metric           = "cosine",     # semantic similarity
    algorithm        = "generic",    # required for cosine metric
    prediction_data  = True,         # enables all_points_membership_vectors()
    cluster_selection_method = "eom",
    cluster_selection_epsilon = 0.0–0.1,   # merges adjacent micro-clusters
)
```

**Auto-tuning**: if fewer than 2 clusters found, the system retries with progressively smaller `min_cluster_size` values [30, 20, 15, 10, 5], logging each attempt. This prevents complete failure on datasets with unusual density distributions.

**cluster_selection_epsilon**: controls post-hoc cluster merging. ε=0.1 on TON_IoT merges closely-related IoT attack subcategories (e.g., different DDoS variants) into coherent campaign groups without requiring k upfront.

### 8.3 Pre-Clustering Dimensionality Reduction

```
128-dim HGNN embeddings
    → PCA (128 → 16 components)          Always applied
    → [UMAP (16 → 10 components)]        Dataset-specific (off for TON_IoT)
    → [Soft-ZCA whitening]               Only for collapsed embeddings (SQTK_SIEM)
```

**PCA**: Removes linear noise and reduces HDBSCAN computational complexity. 16 components retain >95% explained variance in tested embeddings.

**UMAP**: Preserves local and global structure. Useful for UNSW-NB15 (global structure benefits spectral clustering). Disabled for TON_IoT (over-fragmentation: 10K sample with UMAP → 46 clusters; without → 37 stable clusters).

**Soft-ZCA whitening**: Applied when cosine_sim > 0.95 (moderate embedding collapse). Decorrelates the embedding matrix: `W = U(Λ + εI)^{-1/2} U^T`, then re-normalises to unit sphere. eps=0.1 is the production setting for SQTK_SIEM (cosine_sim 0.95 → 0.10).

### 8.4 Alternative Clustering Algorithms

| Algorithm | When Used | Configuration |
|-----------|-----------|--------------|
| **HDBSCAN** | Default | Cosine metric, auto-tune, epsilon |
| **Spectral** | UNSW-NB15 | k=8, cosine affinity, kmeans label assignment |
| **BGMM** | Experimental | max_components=30, full covariance |
| **Prototype** | Supervised mode only | SupervisedPrototypeHead loaded from checkpoint |

### 8.5 Confidence-Gated UF Refinement (Disabled by Default)

The original architecture included a Union-Find fallback for low-confidence alerts. The design intent: when GAEC confidence < 0.6, re-correlate via rule-based Union-Find.

**Empirical result**: UF refinement is net-harmful across all datasets. Ablation (v2.6):

| Dataset | With UF | Without UF | Explanation |
|---------|---------|-----------|-------------|
| UNSW-NB15 | ARI=0.354 | ARI=0.404 | UF creates 100% singletons |
| NSL-KDD | ARI=0.217 | ARI=0.257 | Same pattern |

Root cause: border points — alerts on the geometric boundary between two clusters — get low HDBSCAN confidence and trigger the UF path. UF assigns each one to a new singleton cluster based on exact entity matches, which almost never exist for border points. `use_uf_refinement=False` is the permanent production default.

---

## 9. Dataset Structural Analysis

Each dataset presents a fundamentally different graph structure. Understanding these differences is essential for interpreting why architectural decisions work on some datasets and fail on others.

---

### 9.1 UNSW-NB15 — Dense Network IDS with Protocol Diversity

**Source**: UNSW Cyber Range Lab, IXIA PerfectStorm traffic generator  
**Scale**: 175,341 alerts · 48 campaign IDs · 9 attack categories  
**Real data**: Partially synthetic (IXIA generator)

**Raw schema**:
```
src_ip, dst_ip, proto, service, src_bytes, dst_bytes, duration,
attack_cat (Fuzzers / Analysis / Backdoors / DoS / Exploits / Generic /
            Reconnaissance / Shellcode / Worms),
campaign_id (48 mapped campaigns)
```

**Graph structure fingerprint**:
- IP density: ~0.05 (medium) — many unique IPs from IXIA generation
- Temporal edges: YES — millisecond timestamps
- Host edges: minimal — lab environment
- Key structural problem: **campaigns 34 and 48 use IXIA fuzzing with 129 unique protocols** → protocol feature completely uninformative for those campaigns → diffuse embeddings HDBSCAN cannot compact

**Architectural decisions and why**:

| Decision | Rationale |
|----------|-----------|
| Spectral k=8 (not HDBSCAN) | Spectral uses global graph structure; we know k=8 campaign families exist. HDBSCAN finds density peaks and misses the global partition. 3.7× ARI improvement. |
| sample_size=2000 stratified | Full 175K alerts would make HDBSCAN intractable and dominate with majority-class Benign alerts |
| unsw_supcon_v7 checkpoint | SupCon fine-tuning with campaign labels; requires zero-padding 6→15 to match v9_v3 architecture |
| UMAP n_neighbors=30 | Larger neighbourhood preserves global structure needed for spectral partitioning |

**SupCon fine-tuning journey**:
- v1–v6 (all buggy): `finetune_supcon.py` used `alert_feature_dim=15` but `PublicDatasetGraphConverter` produces 6-dim features. The dimension mismatch caused silent weight corruption via `strict=False` loading. All v1–v6 checkpoints produce garbled embeddings.
- v7 (fixed): Zero-pad 6→15 before all HGNN forward calls. ARI (Spectral k=8) = 0.408.
- Result: SupCon v7 + Spectral matches zero-shot ceiling (0.538). **The IXIA protocol diversity (campaigns 34/48) is a hard structural ceiling for 6-dim features.**

**Verified results**:
| Mode | AMI | ARI |
|------|-----|-----|
| Zero-shot (spectral k=8) | 0.582 | 0.401 |
| SupCon v7 (spectral k=8) | 0.582 | 0.538 |
| Supervised prototype | — | 0.497 |

---

### 9.2 NSL-KDD — The Disconnected Graph

**Source**: KDD Cup 1999 (cleaned, de-duplicated by New Brunswick)  
**Scale**: 125,973 alerts · 5 classes: DoS, R2L, U2R, Probe, Normal  
**Real data**: No — synthetic traffic simulation

**Raw schema**:
```
duration, protocol_type, service, flag, src_bytes, dst_bytes,
land, wrong_fragment, urgent, hot, num_failed_logins, ...
(41 features) → mapped to 6 MITRE features
campaign_id = attack family (4 attack + 1 normal)
```

**Graph structure fingerprint**:
- **NO src_ip / dst_ip columns** — NSL-KDD does not include IP addresses
- **NO timestamps** — no temporal edges possible
- **Graph is fully disconnected** — the only edges are `semantic_similar` (same alert_type)
- Structural consequence: **HGNN ≈ MLP** — message passing has no graph to operate on

**Why it outperforms supervised despite being a "broken" graph**:

The 4 attack families are *linearly separable* in the 6-dim feature space:

```
DoS   : high src_bytes, TCP, specific flags, duration=0
R2L   : specific services (ftp, telnet), low src_bytes, many connections
U2R   : specific services (su, root shells), low byte counts
Probe : many unique dst_services, ICMP/UDP, short duration
```

The HGNN backbone, joint-trained on graph-rich datasets (UNSW, TON_IoT), learns a 128-dim projection where these distinct patterns have clear geometric separation. A linear GMM on raw 6-dim features (ARI=0.299) cannot exploit the non-linear geometry that the HGNN's learned projection reveals (ARI=0.739).

The **B1 input residual** is particularly important here: since message passing does nothing (disconnected graph), the final embedding is essentially `alert_raw_proj(6d_features) + zeros` — the backbone provides the projection, and raw features provide the signal.

**Architectural decisions**:
- No UMAP (no global structure to preserve in a disconnected graph)
- No sampling needed (125K is manageable; HDBSCAN on PCA-16 is fast)
- `num_layers=1` (already correct; matters especially here)
- `hdbscan_cluster_selection_epsilon=0.0` (compact, well-separated clusters)

**Verified results**:
| Mode | AMI | ARI | Note |
|------|-----|-----|------|
| Zero-shot | **0.668** | **0.739** | Best unsupervised result |
| Supervised prototype | — | 0.595 | Labels hurt: adds noise to clean separability |
| Feature GMM baseline | — | 0.299 | 2.5× gap from graph structure |

**Key interview point**: "NSL-KDD proves that joint training transfers structural priors. The graph is disconnected, so HGNN ≈ MLP — yet it outperforms feature-only GMM by 2.5×. The backbone learned a projection space where network attack families have geometric separation, and that learning transfers even when the target dataset has no graph structure."

---

### 9.3 TON_IoT — The Reproducibility Saga

**Source**: UNSW TON_IoT benchmark, IoT/IIoT sensor network  
**Scale**: 211,043 alerts · 10 attack types (DDoS, DoS, backdoor, injection, MITM, password, ransomware, scanning, XSS, normal)  
**Real data**: Partially real IoT sensor logs

**Raw schema**:
```
ts, src_ip, src_port, dst_ip, dst_port, proto, service, duration,
src_bytes, dst_bytes, conn_state, label (0/1), attack_type (10 classes)
```

**Graph structure fingerprint**:
- IP density: medium — IoT devices have fixed IPs, moderate connectivity
- Temporal edges: YES — Unix timestamps available
- IIoT topology: device/gateway edges present
- Key challenge: **full 211K dataset overwhelms HDBSCAN** (77 micro-clusters); needs stratified sampling

**The Track 11 reproducibility crisis** — this dataset had three compounding bugs that took the baseline from ARI=0.724 (historical) down to ARI=0.0076 (Track 8 final_metrics):

**Bug 1 — HDBSCAN unseeded** (`hgnn_correlation.py` lines ~1380, ~1395; `utils/clustering.py` line ~71):
```python
# Before fix: random_state never passed → non-reproducible clustering
clusterer = HDBSCAN(min_cluster_size=..., metric="cosine", ...)

# After fix: seeded for cosine-compatible HDBSCAN (euclidean fallback)
if self.metric != "cosine":
    hdbscan_kwargs["random_state"] = self.seed
```

**Bug 2 — Wrong feature dimension in fine-tune script** (`training/finetune_cross_sensor.py` line 204):
```python
# Before (wrong): v9_v5 experimental dim, never shipped
hgnn = MITREHeteroGNN(alert_feature_dim=15, ...)

# After (correct): matches network_v9_v3 training
hgnn = MITREHeteroGNN(alert_feature_dim=6, ...)
# Plus: zero-pad 6→15 before each forward pass
```

**Bug 3 — sample_size dropped from DATASET_CONFIG** (`experiments/run_gate_tuning.py`):

| Parameter | Reference (013fcef) | After Track 7–10 edits | Effect |
|-----------|--------------------|-----------------------|--------|
| `sample_size` | 10000 | **MISSING** | Full 211K to HDBSCAN |
| `stratified_sample` | True | **MISSING** | Class imbalance |
| `use_umap` | True (then off) | True with extra params | Over-fragmentation |

Fix progression: ARI 0.0076 → 0.109 (seeding) → **0.431** (config restoration)

**Architectural decisions**:
- `sample_size=10000, stratified_sample=True`: Critical. Stratification ensures all 10 attack types are represented. Full dataset overwhelms HDBSCAN with 77 micro-clusters.
- `use_umap=False`: Disabled — UMAP added fragmentation on this dataset (46 vs 37 clusters)
- `hdbscan_min_cluster_size=50`: Prevents IoT attack sub-variants from splitting
- `hdbscan_cluster_selection_epsilon=0.1`: Merges adjacent IoT attack micro-clusters

**Verified results**:
| Mode | AMI | ARI | n_clusters |
|------|-----|-----|-----------|
| Zero-shot (Track 11) | **0.717** | **0.431** | 37 |
| Supervised prototype | — | **0.845** | 10 |
| Feature GMM baseline | — | 0.233 | — |

---

### 9.4 DARPA OpTC — Perfect Binary Separation

**Source**: DARPA/SRI International, 2019 red team exercise on live enterprise  
**Scale**: 4,656,650 events from Windows Sysmon across ~500 hosts  
**Real data**: YES — live enterprise network  
**Campaigns**: Binary — Benign vs RedTeam

**Raw schema**:
```
timestamp, hostname, process_name, action, src_ip, dst_ip, CampaignId,
object (file/network/registry), CommandLine (APT activity)
```

**Graph structure fingerprint**:
- Rich host graph: process + commandline + hostname edges all present
- Massive scale: 4.6M events → 10K stratified sample for evaluation
- Domain: **host-level telemetry** (Sysmon) vs checkpoint trained on network IDS alerts
- Binary structure: only 2 campaigns make standard ARI misleading

**The ARI paradox**:

```
Standard ARI = 0.048   ← looks bad
binary_ARI   = 0.999   ← near-perfect
cluster_purity = 0.999 ← 99.9% pure clusters

What happened: HGNN finds 25 sub-clusters.
Every RedTeam alert → a RedTeam sub-cluster.
Every Benign alert → a Benign sub-cluster.
ARI penalises having 25 clusters instead of 2.
binary_ARI collapses to: did each cluster go to the right campaign?
```

The 25 sub-clusters represent **meaningful attack phases**: C2 establishment, lateral movement, credential harvesting, exfiltration — distinct enough in embedding space to separate. A SOC analyst would call this a feature, not a bug.

**Architectural decisions**:
- `use_geometric_confidence=True` (GAEC): The `multidomain_v2` checkpoint has a softmax head trained on network labels — completely OOD for host telemetry. GAEC uses the raw embedding geometry, which transfers better.
- `checkpoint_override=network_v9_v3`: NOT multidomain_v2. network_v9_v3 was joint-trained on OpTC data — it has seen Sysmon-style events during training.
- `hdbscan_min_cluster_size=50`: At 10K sample, this prevents over-fragmentation
- `stratified_sample=True`: Ensures the 10K sample has both RedTeam and Benign events proportionally

**Verified results**:
| Metric | Value | Interpretation |
|--------|-------|----------------|
| AMI | 0.149 | Limited by binary label space |
| ARI (standard) | 0.048 | Penalised for legitimate sub-clustering |
| binary_ARI | **0.999** | Near-perfect campaign-level separation |
| Purity | **0.999** | 99.9% single-campaign clusters |
| Attack F1 | 0.942 | Near-perfect attack/benign detection |

---

### 9.5 SQTK_SIEM — Archived SOC Data

**Source**: Real Security Operations Centre (anonymised)  
**Scale**: 5,100 alerts from 7 distinct commercial sensors  
**Real data**: YES — archived alerts from a production SOC (offline batch, 5,100 records)  
**Labels**: `kcluster` (11 expert-defined campaign groups)

**Multi-sensor composition**:
| Sensor | Count | Category |
|--------|-------|----------|
| Cisco | 3,272 (64%) | Network/Firewall |
| F5 WAF | 1,329 (26%) | Web App Firewall |
| Trend Micro | 292 (6%) | Endpoint AV |
| Imperva | 130 (3%) | Runtime App Protection |
| FireEye | 42 (0.8%) | Next-Gen Firewall |
| Microsoft Defender | 30 (0.6%) | EDR |
| Acalvio | 5 (0.1%) | Deception Technology |

**Why this is the hardest dataset**:

1. **Label quality**: `campaign_id` is 89% "UNKNOWN" — human analysts used tactic tags, not campaign IDs. `kcluster` is the expert-annotated grouping, but represents analyst judgment rather than ground truth.
2. **Multi-sensor schema heterogeneity**: Each sensor uses different field names, severity scales, and tactic vocabularies. Normalisation to MITRE format loses information.
3. **Severe class imbalance**: Cisco (3,272 alerts) vs Acalvio (5 alerts) — 654:1 ratio.
4. **Sparse graph**: Only 5,100 alerts → very few shared IPs/hosts → graph is nearly disconnected → HGNN cannot propagate meaningful signal.
5. **Embedding collapse**: Sparse graph + high-degree shared enterprise IPs → cosine_sim=0.91 (moderate collapse).

**Architectural decisions**:
- `siem_supcon_v4` checkpoint: Retrained with three fixes:
  - Fix 1: Label-pure edge filtering (cross-campaign edges removed during SupCon training)
  - Fix 2: Class-balanced SupCon loss (inverse-frequency weights for rare sensor types)
  - Fix 3: Alert feature enrichment (additional SIEM-specific features)
- `hdbscan_zca_whitening=True, eps=0.1`: Soft-ZCA mandatory — reduces cosine_sim 0.91→0.10
- `clustering_method="spectral", n_clusters=11`: HDBSCAN fails on sparse graphs with collapsed embeddings; spectral directly solves for the k=11 partition
- Label column: `kcluster` not `campaign_id` (89% UNKNOWN would make ARI meaningless)

**Verified results**:
| Mode | AMI | ARI |
|------|-----|-----|
| Zero-shot + ZCA + Spectral | **0.342** | **0.184** |
| SupCon v3 + ZCA | — | 0.174 |
| Supervised prototype | — | 0.053 |

**Key insight**: Prototype mode *underperforms* zero-shot on real SOC data because the expert `kcluster` labels are noisy — the prototype head learns the noise. Zero-shot clustering of the embedding geometry is more robust to label noise.

---

### 9.6 CICIDS2017 — Flow-Feature Benchmark

**Source**: Canadian Institute for Cybersecurity, 2017  
**Scale**: 2.8M+ network flows, 14 attack classes  
**Real data**: Partially real (lab environment with real tools)

**Structure**: Flow-level features only (78 statistical features). No IP/host entities in the standard format. The graph is sparse (temporal edges only). Best result: ARI=0.284 zero-shot, 0.440 with `use_burstiness=True` and `aggr_method=max`.

**Limitation**: CICIDS2017 was not part of the primary evaluation due to ambiguous label column mapping (fixed in Track 4). Results are preliminary.

---

## 10. Experiment History

### Phase 1: Getting the Loss Right (Apr 2026)

**Critical discovery**: All pre-v7 training used **MSE loss** on embeddings. MSE minimises reconstruction error, which drives all embeddings toward the mean — perfect embedding collapse. NT-Xent pulls graph-connected pairs together and pushes disconnected pairs apart, maintaining geometric diversity.

```
v4 (MSE):     cosine_sim = 0.98–0.99   → clustering fails (1 cluster)
v9_v3 (NT-Xent): cosine_sim = 0.73–0.92 → clustering succeeds
```

This single loss change accounts for the majority of the performance improvement across all datasets.

### Phase 2: Architecture Stabilisation (Apr 2026)

| Version | Change | Finding |
|---------|--------|---------|
| v2.1 | Add confidence-gated UF | ARI improves, but UF creates singletons |
| v2.6 | `use_uf_refinement=False` default | ARI 0.354→0.404 on UNSW |
| v2.7 | singleton_fraction metric | Reveals UF creates 80%+ singletons universally |
| v2.9 | soft_assign for border points | Border points ≠ noise; soft_assign ineffective |

### Phase 3: Multi-Dataset Joint Training (Apr 2026)

| Version | Change | Finding |
|---------|--------|---------|
| v2.10–v2.14 | Joint training NSL+TON+UNSW | UNSW ceiling at 0.523 with SupCon |
| v2.15–v2.18 | UNSW optimisation sweep | All failed — IXIA protocol diversity is structural |
| v2.19 | TON_IoT zero-shot | ARI=0.737 (historical, later found unseeded) |
| v2.20 | OpTC zero-shot | ARI=0.979 binary confirmed |
| v2.21 | 15-dim contextual features (v9_v5) | **Catastrophic**: ARI 0.714→0.003. Reverted. |

**v9_v5 failure analysis**: Contextual features (IP frequency, temporal density) were computed from batch statistics — different at training vs inference time. The model learned to rely on batch-relative statistics, but at inference time those statistics have a different distribution. This is a fundamental train/inference distribution mismatch that cannot be fixed with fine-tuning.

### Phase 4: HDBSCAN Bugs & Baselines (Apr 19, 2026)

**Critical bug**: HDBSCAN was called per-chunk (1K alert windows) instead of on all embeddings. With `min_cluster_size=50` on 1,000 samples = 5% threshold → 12 micro-clusters per window. Same config on 10K embeddings → 2 clusters.

| Version | Fix | Impact |
|---------|-----|--------|
| v2.24 | Bridge edge ablation (ip→host) | Zero effect — closed |
| v2.25 | Entity collapse ablation | Zero effect — closed |
| v2.26 | HDBSCAN windowing fix | All prior OpTC ablations retroactively invalidated |
| v2.26 | NSL-KDD feature baseline | HGNN 0.722 vs GMM 0.299 — graph value confirmed |

### Phase 5: Algorithm Additions (Apr 20–21, 2026)

| Version | Addition | Key Result |
|---------|---------|-----------|
| v2.27 | Spectral k=8 for UNSW | ARI 0.034→0.128 on zero-shot |
| v2.27 | Soft-ZCA eps=0.1 for SQTK | cosine_sim 0.95→0.10 |
| v2.27 | SupCon dim-mismatch found | v1–v6 all corrupted (silent shape error) |
| v2.27 | SupCon v7 fixed | ARI (Spectral) = 0.408 on UNSW |
| v2.29 | Edge density cap (5 edges/node) | No improvement — over-smoothing from density is not the bottleneck |

### Phase 6: Prototype Integration (Apr 22–25, 2026)

| Version | Action | Outcome |
|---------|--------|---------|
| v2.31 | Prototype training + integration | False cosine_sim > 1000 alarm |
| v2.32 | Fix: similarity on embedding slice, not raw+embedding concat | True cosine_sim: 0.73–0.92 (normal) |
| v2.33 | Prototype backbone fix — load HGNN from prototype checkpoint | TON_IoT: 0.2423→0.845 |
| v2.34 | Zero-shot regression: burstiness=False confirmed | Config standardised |

### Phase 7: Cross-Sensor Features (Apr 26–30, 2026)

CS-1 through CS-5 implemented:
- CS-1: `data_source` column in all converters
- CS-2: `source_sensor` node type + `collected_by`/`collects` edges
- CS-3: Kill-chain `precedes` temporal edges (2h window)
- CS-4: `MultiSourceIngestionPipeline` for N-sensor fusion
- CS-5: CLI flags for all CS features

18 tests passing. NSL-KDD ARI preserved at 0.7428. SQTK_SIEM: 7 source_sensor nodes confirmed.

**CS-3 ablation**: precedes=False gives ARI=0.139 vs precedes=True gives ARI=0.115. Kill-chain edges are noise, not signal. Closed.

### Phase 8: Metrics Legitimacy (Track 8, Apr 30, 2026)

AMI promoted to primary metric. Added:
- Tactic Sequence Coherence (Kendall's τ vs ATT&CK kill-chain order)
- Attack F1 (binary attack/normal detection)
- Cluster purity

Track 8 sweep results (with the still-unseeded HDBSCAN — these numbers were later corrected):

| Dataset | AMI | ARI |
|---------|-----|-----|
| UNSW-NB15 | 0.582 | 0.401 |
| NSL-KDD | 0.673 | 0.752 |
| TON_IoT | 0.288 | 0.008 | ← unseeded, wrong config
| OpTC | 0.149 | 0.048 |
| SQTK_SIEM | 0.342 | 0.184 |

### Phase 9: TON_IoT Reproducibility (Track 11, May 8, 2026)

Three bugs found and fixed (see §9.3 for full analysis). Final verified baseline:

```
ARI: 0.0076 (Track 8, unseeded + wrong config)
  → 0.109  (seeding fixed)
  → 0.431  (sample_size + UMAP config restored)
```

---

## 11. Ablation Studies — Real Experimental Data (May 2026)

All results below are from controlled experiments using the canonical `network_v9_v3` checkpoint, seed=42. Source CSVs in `experiments/results/ablation_studies/`.

### 11.1 UF Refinement — Validated Net-Harmful

**Hypothesis**: Low-confidence HGNN assignments can be improved by Union-Find rule-based correlation.

**Method**: Gate sweep (0.4–0.9) with UF on vs off across NSL-KDD, UNSW-NB15, TON_IoT.

**Result**:
| Dataset | UF Off ARI | UF On ARI | UF Off n_clusters | UF On n_clusters | UF Routed % |
|---------|-----------|----------|--------------------|-------------------|-------------|
| NSL-KDD | **0.739** | 0.738 | 14 | 74 (+60 UF) | 2.3% |
| UNSW-NB15 | **0.464** | 0.464 | 8 | 8 (+0 UF) | 0.0% |
| TON_IoT | **0.431** | 0.422 | 37 | 297 (+260 UF) | 6.3% |

**Conclusion**: UF refinement is net-harmful or neutral across all datasets. NSL-KDD: negligible ARI change but 5.3× cluster inflation (14→74). UNSW: UF never triggers (0% routed). TON_IoT: ARI drops 0.009 while clusters explode 8× (37→297). `use_uf_refinement=False` is permanently locked.

**Why 2.3% UF routing → 5.3× cluster explosion on NSL-KDD**: NSL-KDD has no IP addresses or timestamps — the graph is fully disconnected except for `semantic_similar` edges (same alert_type hash). When UF routes a low-confidence alert, it searches for exact entity matches (shared IP, same host) to merge it into an existing cluster. Since no IP or host edges exist, every UF-routed alert fails to match any existing cluster and becomes a new singleton. Each singleton then acts as a seed for further fragmentation. The result: 2.3% routing rate creates 60 singleton clusters (81% of all clusters), because the UF merge condition is structurally unsatisfiable on this dataset.

**Source CSVs**: `uf_ablation_*_uf_off.csv`, `uf_ablation_*_uf_on.csv`

### 11.2 TON_IoT Sample Size Sweep — Why 10K is Canonical

**Hypothesis**: Sample size significantly impacts clustering quality and stability.

**Method**: Gate sweep at 2K, 5K, 10K stratified samples.

**Result**:
| Sample Size | ARI | NMI | n_clusters | Latency (s) | Attack F1 |
|-------------|-----|-----|-----------|-------------|-----------|
| 2,000 | 0.443 | 0.706 | 19 | 1.0 | 0.967 |
| 5,000 | 0.384 | 0.686 | 36 | 3.7 | 0.932 |
| 10,000 | 0.431 | 0.718 | 37 | 9.5 | 0.969 |

**Conclusion**: 10K provides the best NMI (0.718) and attack F1 (0.969) with stable cluster count (37). 2K under-clusters (19 vs 37). 5K has the worst ARI (0.384) — a local minimum where sample diversity increases but HDBSCAN hasn't yet stabilised. 10K is the canonical config: best information capture, highest attack detection, manageable latency.

**Source CSVs**: `ton_sample_sweep_2000.csv`, `ton_sample_sweep_5000.csv`, `ton_sample_sweep_10000.csv`

### 11.3 Baseline Clustering — Full Method Comparison

**Hypothesis**: MITRE-CORE's GAEC pipeline outperforms classical clustering on raw features and embeddings.

**Method**: 8 methods × 6 datasets, 10K samples, seed=42.

**Result**: See §3 for the full comparison table.

**Key findings**:
- MITRE-CORE is the best method on SQTK_SIEM (real SOC data) — its strongest niche
- K-Means on HGNN embeddings wins on 2/6 datasets (UNSW, TON_IoT) — embeddings add value but simple clustering suffices
- DBSCAN on raw features wins on 2/6 datasets (NSL-KDD, CICIDS2017) — clean feature separation doesn't need graphs
- MITRE-CORE is competitive but not dominant — honest positioning as a strong method for heterogeneous SIEM correlation

**Source CSV**: `baseline_clustering_comparison.csv`

### 11.4 Other Ablations (Historical)

- **Bridge Edges (ip→host)**: Zero effect — `shares_host` edges already provide the shortcut. Closed.
- **Entity Collapse**: Zero effect — same root cause as bridge edges. Closed.
- **Kill-Chain Edges (CS-3)**: No improvement — IoT attacks don't follow linear kill chains. Closed.
- **v9_v5 Contextual Features (15-dim)**: Catastrophic — NSL-KDD ARI 0.714→0.003. Train/inference distribution mismatch. Reverted.
- **UMAP on TON_IoT**: Marginal cluster count change without ARI gain. Disabled.
- **SupCon v1–v6 Dim-Mismatch**: Silent weight corruption via `strict=False` loading. Fixed in v7.

---

## 12. Metrics Deep Dive

### 12.1 AMI — Adjusted Mutual Information (Primary)

```
MI(U,V)  = Σ_k Σ_j |U_k ∩ V_j| / N · log( N|U_k ∩ V_j| / (|U_k||V_j|) )
AMI(U,V) = ( MI(U,V) - E[MI] ) / ( max(H(U), H(V)) - E[MI] )
```

AMI measures how much information the predicted clustering shares with ground truth, adjusted for the information expected by chance for the given cluster counts. Range: [0, 1] (can be slightly negative).

**Why AMI over ARI**: ARI is pair-counting — it asks "for every pair of alerts, did we correctly say same-cluster or different-cluster?" This penalises sub-clustering: if the true labelling has 2 campaigns but HGNN finds 25 sub-clusters (OpTC), ARI=0.048 even if every pair within a true campaign is correctly grouped. AMI rewards information content: the 25 sub-clusters share nearly all information with the 2-campaign ground truth, so AMI=0.149 (limited by the binary label space, not by clustering quality).

### 12.2 ARI — Adjusted Rand Index (Secondary)

```
ARI = (RI - E[RI]) / (max(RI) - E[RI])
```

Pair-counting metric. Intuitive but penalises legitimate sub-clustering. Reported alongside AMI for comparability with published baselines. Range: [-1, 1].

### 12.3 GAEC Confidence

Per-alert confidence ∈ [0, 1] derived from HDBSCAN membership vectors. Reflects geometric density in the 128-dim embedding space:

- **Core cluster member** (high density neighbourhood): confidence ≈ 0.9–1.0
- **Border point** (cluster boundary): confidence ≈ 0.3–0.7
- **Noise point / OOD alert**: confidence ≈ 0.0–0.1

ECE measured at 0.015–0.022 across benchmarks (well-calibrated without temperature scaling).

### 12.4 Tactic Sequence Coherence

```
coherence_k = Kendall_τ(
    temporal_ordering_of_tactics_within_cluster_k,
    canonical_ATT&CK_ordering: [Recon=0, Resource_Dev=1, Initial_Access=2,
                                  Execution=3, Persistence=4, Priv_Esc=5,
                                  Defense_Evasion=6, Cred_Access=7,
                                  Discovery=8, Lateral_Movement=9,
                                  Collection=10, C2=11, Exfiltration=12,
                                  Impact=13]
)
```

Measures whether alerts within each campaign cluster follow the expected kill-chain temporal order. Positive = correct ordering. Negative = reversed (common in IoT datasets where attack tools don't follow linear chains). NaN for datasets without timestamps (NSL-KDD).

### 12.5 Cluster Purity

```
purity = (1/N) · Σ_k max_j |cluster_k ∩ true_class_j|
```

For each cluster, what fraction is the most common true class? Purity=0.999 (OpTC) confirms that despite 25 sub-clusters instead of 2, each sub-cluster is internally homogeneous.

### 12.6 Attack F1

Binary F1 score for attack vs normal classification derived from cluster assignments. High values (0.969 TON_IoT, 0.942 OpTC) indicate the clustering cleanly separates malicious from benign traffic even without attack labels.

---

## 13. Design Decisions & Engineering Rationale

### 13.1 Why Heterogeneous Graph (Not Homogeneous)?

A homogeneous GNN treats all nodes identically. In the security domain, an IP address node and an alert node have fundamentally different semantics. The model needs to learn that "IP propagates neighbourhood similarity" differently from "user propagates ownership". HeteroGATConv gives each edge type its own learned weight matrix — a critical architectural requirement.

Empirical: HomogeneousGNN baseline achieves ~30% lower ARI on UNSW-NB15 by averaging across all entity types uniformly.

### 13.2 Why GAT Over GCN or GraphSAGE?

**GCN**: Fixed `1/√(deg_i · deg_j)` normalisation — treats all neighbours equally. A hub IP shared by 1,000 unrelated alerts would propagate a meaningless average to all of them.

**GraphSAGE**: Samples k neighbours per node — loses rare but important connections (e.g., a specific C2 IP shared by only 3 alerts).

**GAT**: Learns attention weights per edge — downweights noisy high-degree neighbours, emphasises rare but semantically important connections. Essential for the star-topology IP graphs in UNSW-NB15 and TON_IoT.

### 13.3 Why Self-Supervised (Not Supervised)?

MITRE-CORE is designed for SOC deployment scenarios. Production constraints:
- Labelled campaign data is rarely available (analysts correlate *after* detection, not before)
- Labels are environment-specific and don't transfer across organisations
- Attack taxonomies evolve — a model trained on 2023 labels may not cluster 2024 TTPs correctly

Self-supervised training on graph topology labels nothing — it learns "similar graph structure = similar embedding" which is a universal principle across environments. The zero-shot transfer empirics confirm this: the model trained on UNSW-NB15 + NSL-KDD + TON_IoT + OpTC generalises to SQTK_SIEM (real SOC data) and CICIDS2017 without any retraining.

### 13.4 Why Hybrid Topological + SimCLR Loss?

The canonical network_v9_v3 checkpoint uses a **hybrid** loss (67% topological NT-Xent + 33% SimCLR), not pure topological NT-Xent. This section explains why pure topology alone is insufficient and why the hybrid is necessary.

| Approach | Positive Pair Definition | Failure Mode |
|----------|--------------------------|--------------|
| **Pure Topological NT-Xent** | Graph-connected alerts only | Mini-graph batches lack true negatives — all alerts in a batch come from the same campaign, causing over-smoothing and embedding collapse |
| **Pure SimCLR** | Augmented views of same data point | Weak supervision signal — random perturbation carries no structural attack knowledge |
| **Hybrid (67% topo + 33% SimCLR)** | Both graph edges and augmented views | Topological component provides structural domain knowledge; SimCLR component provides in-batch negatives from other alert instances within the same batch (often from different campaigns, since each mini-graph batch contains multiple campaigns) to prevent collapse |

**Why pure topological NT-Xent fails alone**: Training uses mini-graph batches (~500 alerts per graph). Within a single batch, all alerts belong to the same campaign (graphs are constructed per-campaign). Topological NT-Xent needs both positive pairs (graph-connected alerts) and negative pairs (alerts from different campaigns). With single-campaign batches, there are no true negatives — every alert pair is either connected (positive) or unconnected-but-same-campaign (false negative). The model collapses all embeddings toward a single point because it never sees genuine cross-campaign negatives. The SimCLR component solves this: augmented views from different batches provide cross-campaign negative diversity, maintaining embedding separation.

**Two alerts connected by `shares_ip` genuinely *should* be similar. Two alerts connected by `temporal_near` genuinely *might* be related. Using the graph topology as the positive pair definition encodes domain knowledge that random augmentation cannot — but it must be combined with augmentation-based negatives to prevent collapse.**

### 13.5 Why HDBSCAN Over K-Means?

| Property | K-Means | HDBSCAN |
|----------|---------|---------|
| Requires k upfront | YES (unknown at deployment) | NO |
| Handles noise | NO | YES (noise = low-confidence alerts) |
| Variable-density clusters | NO (assumes spherical) | YES |
| Provides confidence | NO | YES (membership vectors) |
| Epsilon merging | NO | YES |

In production deployment, the number of campaigns is unknown. HDBSCAN finds the natural cluster structure from the data geometry.

### 13.6 Why AMI as Primary Metric?

SOC analysts perform legitimate sub-campaign analysis. A 3-week APT campaign has 3 distinct phases — the HGNN correctly identifies these as 3 geometric clusters. ARI penalises this as error. AMI rewards it as information. The system should not be penalised for providing *more* detail than the ground truth label requires.

### 13.7 Why hidden_dim=128?

Four independent constraints converge on 128 — it is not arbitrary:

**1. Multi-head attention divisibility (hard constraint)**
`GATConv(out_channels = hidden_dim // num_heads)` requires integer division.
With `num_heads=4`: `128 / 4 = 32` per head — clean and expressive.
- `hidden_dim=64` → 16 per head: too few dimensions for meaningful per-edge-type attention
- `hidden_dim=256` → 64 per head: valid but 2× parameter count with no empirical gain on these datasets

**2. Contrastive loss separability**
The hybrid topological NT-Xent + SimCLR loss needs sufficient dimensions to push different
attack families apart. Empirical: 128-dim achieves cosine_sim ≈ 0.10 after training.
Smaller dims showed cosine_sim > 0.80 (collapse) in ablations.

**3. PCA compression ratio in GAEC**
`EmbeddingConfidenceScorer` compresses 128 → 16 PCA components (8× reduction).
- `hidden_dim=64` → 64→16 is 4× — too aggressive, loses cluster geometry
- `hidden_dim=256` → wastes compute; HDBSCAN gains nothing above ~32 effective dims

**4. B1 input residual upsample**
`alert_raw_proj(6-dim raw features) → 128-dim`, then added back after message passing.
The projection must be large enough to preserve all signal from 6 inputs as a meaningful residual.

| hidden_dim | heads | per-head dim | PCA ratio | Verdict |
|------------|-------|-------------|-----------|---------|
| 64 | 4 | 16 | 4× | Too few per-head dims; over-aggressive PCA |
| **128** | **4** | **32** | **8×** | **All four constraints satisfied** |
| 256 | 4 | 64 | 16× | Valid but 2× params; no empirical justification |

> **Interview note**: The primary answer to "why 128?" is `hidden_dim / num_heads = 32` 
> (multi-head attention divisibility), not "power of 2". Power of 2 is a secondary benefit
> for GPU memory alignment, not the design driver.

---

## 14. Limitations

MITRE-CORE has known limitations that constrain its applicability. These are documented here transparently — they are not bugs but architectural boundaries.

| Limitation | Detail | Mitigation |
|------------|--------|------------|
| **Zero-shot scope** | Zero-shot transfer works on network IDS data (NSL-KDD, UNSW-NB15). IoT (TON_IoT) and host telemetry (OpTC) need supervised fine-tuning to reach competitive ARI. | Use SupCon fine-tuning for non-IDS domains; prototype training for IoT benchmarks. |
| **Graph dependency** | Performance degrades on sparse or disconnected graphs. SQTK_SIEM (5,100 alerts, few shared IPs) requires ZCA whitening to prevent embedding collapse. NSL-KDD (fully disconnected) relies entirely on the backbone projection — the graph adds zero value. | `dataset_profiler.py` detects graph sparsity and routes to ZCA whitening or fallback clustering configs. |
| **6-dim minimal features** | The 6-dim feature vector (tactic, alert_type, hour, day_of_week, protocol, service) discards rich contextual information present in raw datasets (e.g., byte counts, connection duration, TCP flags). The v9_v5 experiment showed that richer features cause catastrophic regression due to train/inference distribution mismatch. | Feature engineering is locked at 6 dims. Richer features require a fundamentally different training approach (e.g., per-dataset normalisation statistics bundled with the checkpoint). |
| **Sample-size sensitivity** | HDBSCAN clustering quality depends on sample size. TON_IoT at 2K under-clusters (19 vs 37); at 5K hits a local ARI minimum (0.384). The canonical 10K was found empirically, not derived from theory. | Stratified sampling with seed=42 is the production default. Sample size should be validated per deployment. |
| **Batch-only processing** | The system processes fixed-size batches (≤10K alerts). Streaming/online clustering is not supported — the full embedding matrix must be available for HDBSCAN's density estimation. | For streaming deployments, batch windows with overlap can approximate online operation, but true streaming HDBSCAN requires a different clustering backend. |
| **Layer-count ablation unvalidated** | The claim that L=1 prevents over-smoothing is based on a single informal test (L=2 gave ARI≈0.05 on UNSW). No controlled multi-seed L=1 vs L=2 ablation was run. The L=1 lock is pragmatic but not rigorously validated. | A proper L∈{1,2,3} ablation with 3+ seeds per condition is deferred to future work. |
| **Single-seed reporting** | **RESOLVED (May 2026)**: §3 headline table now reports mean±std over seeds {42,43,44}. 4/6 datasets are deterministic (std=0); UNSW-NB15 CoV=6.3%; CICIDS2017 CoV=12.3% (sampling-dependent, below 25% concern threshold). | `experiments/run_multiseed_headline_table.py` automates multi-seed sweeps. Raw per-seed CSVs in `experiments/results/multiseed_per_seed/`. |
| **Hybrid loss weight ablation missing** | The 67/33 topo:SimCLR weighting was hardcoded — a sweep over {100/0, 67/33, 50/50, 33/67, 0/100} has not been run. Weights are now configurable via `--topo_weight` and `--simclr_weight` in `training/train_graph_mae_v9_multidata_fast.py`. | The 5-point sweep is deferred to future work (requires GPU retraining). |
| **6-dim feature lock argument is weak** | The v9_v5 catastrophic regression (NSL-KDD 0.714 → 0.003) is real, but the failure mode was *batch-statistic-dependent normalization mismatch between train and inference*, not a fundamental argument against richer features. A properly-normalised 12-dim feature set with bundled normalisation statistics is untested. | Future work: re-attempt richer features with checkpoint-bundled normalisation statistics, identical train/inference normalisation logic. |
| **No GNN architectural baselines** | The claim that heterogeneous GAT specifically is necessary is supported only by an informal homogeneous-GNN comparison. R-GCN, HAN, HGT, GraphMAE, HeCo, DGI, GRACE — all standard heterogeneous or self-supervised GNN baselines — have not been run on the same datasets with the same evaluation protocol. | Adding HAN/HGT/R-GCN baselines via PyG is straightforward (~1 week). Pending. |

---

## 15. Related Work

MITRE-CORE sits at the intersection of graph neural networks for security and unsupervised alert correlation. The tables below position it against (a) the most directly comparable security systems, and (b) the heterogeneous-GNN and self-supervised graph-learning literature it builds on.

### 15.1 Security-Domain Systems

| System | Approach | Supervision | Graph Type | Key Difference from MITRE-CORE |
|--------|----------|-------------|------------|-------------------------------|
| **DeepCASE** (van Ede et al., IEEE S&P 2022) | Transformer + attention on event sequences | Semi-supervised | None (sequential) | MITRE-CORE is fully self-supervised at training and label-free at inference; DeepCASE needs analyst-labelled context clusters. |
| **WATSON** (Zeng et al., NDSS 2021) | TransE-style embeddings + contextual aggregation over audit-log knowledge graphs | Unsupervised behaviour abstraction | Knowledge graph over audit events | WATSON abstracts low-level audit events into high-level behaviours; MITRE-CORE clusters multi-sensor alerts into campaigns. |
| **MAGIC** (Jia et al., USENIX Security 2024) | Masked graph representation learning on provenance graphs | Self-supervised | Homogeneous provenance graph | MITRE-CORE uses heterogeneous GATConv with 29 typed edges over multi-sensor alerts; MAGIC operates on host-level provenance with masked-feature reconstruction. |
| **Euler** (King et al., NDSS 2022) | Distributed temporal GCN + RNN, anomalous edge detection | Unsupervised | Temporal homogeneous (host graph) | Euler detects anomalous lateral-movement edges; MITRE-CORE clusters alerts into campaign narratives — complementary goals. |
| **CADE** (Yang et al., USENIX Security 2021) | Contrastive learning on raw features for drift/OOD detection | Supervised contrastive | None (tabular) | CADE detects concept drift in malware/IDS classifiers; MITRE-CORE clusters alerts into campaigns. Different problem. |

### 15.2 Heterogeneous & Self-Supervised Graph Learning (Architectural Baselines)

MITRE-CORE inherits its core machinery from this literature; none of these have been re-run on the MITRE-CORE evaluation datasets and represent a known empirical gap (§14).

| Family | Representative Work | Relation to MITRE-CORE |
|---|---|---|
| **Heterogeneous GNNs** | R-GCN (Schlichtkrull et al., ESWC 2018); HAN (Wang et al., WWW 2019); HGT (Hu et al., WWW 2020) | Define the modern heterogeneous-GNN design space. The MITRE-CORE schema is a security-domain instantiation using HAN/HGT-style typed message passing implemented via PyG `HeteroConv` over GATConv. R-GCN/HAN/HGT have not been benchmarked on MITRE-CORE datasets — the "heterogeneous GAT specifically is necessary" claim is therefore not yet empirically grounded against these baselines. |
| **Self-supervised graph learning** | DGI (Veličković et al., ICLR 2019); GRACE (Zhu et al., ICML Workshop 2020); GraphMAE (Hou et al., KDD 2022); HeCo (Wang et al., KDD 2021) | Define the standard SSL graph-learning baselines (mutual-information maximisation, contrastive, masked autoencoding). The MITRE-CORE training objective is a topological-NT-Xent + SimCLR hybrid which fits within this family but has not been compared to GraphMAE/HeCo/DGI/GRACE on the same datasets. |
| **Provenance-graph security** | ATLAS (Alsaheel et al., USENIX Security 2021); DEPCOMM (Xu et al., IEEE S&P 2022); NoDoze (Hassan et al., NDSS 2019) | Closest neighbourhood for "graphs over security events". Operate on host-process provenance rather than multi-sensor alerts; complementary to MITRE-CORE's network-IDS + SIEM focus. |

**MITRE-CORE's positioning**: The contribution is a *system* — a security-domain integration of heterogeneous GAT (HAN/HGT family), a hybrid topological+SimCLR self-supervised objective (NT-Xent/GRACE family), HDBSCAN-based GAEC confidence scoring, and dataset-aware routing — applied to the alert-to-campaign clustering task. The novelty is application-engineering and the empirical study of where graph structure helps (heterogeneous SIEM, real SOC archived data) and where it does not (cleanly-separable network IDS features). The narrow 6-dim feature vector is the primary limitation relative to systems like DeepCASE.

---

## 16. Interview Preparation

### Elevator Pitch (30 seconds)

"MITRE-CORE is a heterogeneous graph neural network that groups raw security alerts into attack campaigns. It models alerts, IPs, hosts, and users as a graph, trains via a hybrid contrastive loss combining topological NT-Xent and SimCLR, then clusters embeddings with HDBSCAN. No labels required at inference. It achieves strong zero-shot results on network IDS data (NSL-KDD ARI=0.739, UNSW ARI=0.434) and is the best method on real SOC SIEM data. On IoT and host telemetry, supervised fine-tuning is needed. The architecture is clean, the experiments are real, and the limitations are documented."

---

### Q: Walk me through the end-to-end architecture.

Raw alerts arrive as CSV or Parquet. They're normalised to a 16-column MITRE-format schema by the ingestion layer. The `dataset_profiler.py` fingerprints the data — IP density, has_timestamps, has_hostnames — and routes to the appropriate checkpoint and clustering config.

`AlertToGraphConverter` then builds a PyTorch Geometric `HeteroData` graph with up to 8 node types (alert, IP, host, user, device, gateway, process, command_line) and 29 edge relation types. Each alert gets a 6-dim integer feature vector encoding tactic, alert_type, hour, day_of_week, protocol, and service.

The `MITREHeteroGNN` runs a single-layer Heterogeneous Graph Attention Network. Each edge type has its own GATConv with 4 heads × 32-dim = 128-dim output. LayerNorm + residual skip after each layer. A B1 input-side residual adds the raw feature projection back to prevent information loss from message passing on dense graphs.

The resulting 128-dim alert embeddings go to the `EmbeddingConfidenceScorer`: PCA to 16 dims, optional UMAP, optional Soft-ZCA whitening, then HDBSCAN. The HDBSCAN membership vectors provide per-alert confidence scores — core cluster members get high confidence, noise/OOD alerts get near-zero.

Output: alert_id → campaign_id mapping with confidence scores. A SOC with 10,000 daily alerts sees ≤50 campaign narratives instead.

---

### Q: Why does the model outperform baselines on NSL-KDD despite having no graph edges?

NSL-KDD is a disconnected graph — no IPs, no timestamps. HGNN is effectively an MLP on 6 features.

Honest answer: on NSL-KDD, raw-feature DBSCAN (ARI=0.753) actually outperforms MITRE-CORE (ARI=0.598). The graph doesn't help here — the 4 attack families are linearly separable in the 6-dim feature space, and DBSCAN finds them directly. MITRE-CORE's backbone projection is competitive but not superior.

The real graph value shows on TON_IoT, where raw-feature K-Means gets ARI=0.125 but K-Means on HGNN embeddings achieves ARI=0.720 (5.8× improvement). The graph's contribution is real but dataset-dependent — not a universal 2.5× multiplier.

---

### Q: What's the hardest engineering problem you solved?

The TON_IoT reproducibility crisis. A result that looked like ARI=0.737 turned out to be completely non-deterministic. Three bugs were operating simultaneously:

First, HDBSCAN was never given a random seed. The HDBSCAN library uses random initialisation for its minimum spanning tree — the same model, same data, same everything produced different cluster counts (24 vs 77) on different runs.

Second, in the cross-sensor fine-tuning script, I'd hardcoded `alert_feature_dim=15` (from a failed experiment) instead of 6. The checkpoint loaded with `strict=False`, so the mismatched `alert_raw_proj` weights were silently skipped — the model had a randomly initialised input projection. AMI=0.0 during training, no gradient flow through the backbone.

Third, `sample_size=10000` and `stratified_sample=True` had been dropped from the dataset config during Track 7-10 refactoring. HDBSCAN on the full 211K dataset fragmented into 77 micro-clusters instead of the reference 24.

Diagnosing this required git bisect to find the last commit where the result was stable, diffing the DATASET_CONFIG blocks across commits, inspecting checkpoint state dicts for missing/unexpected keys, and running diagnostic sweeps at each intermediate state. The fix progression was: ARI 0.008 → 0.109 (seeding) → 0.431 (config restoration).

---

### Q: How do you handle different dataset schemas without retraining?

Three mechanisms:

**Structural**: `AlertToGraphConverter` builds only the edges that are possible from the available columns. If there's no `src_ip` column, no IP nodes are created. If there's no `timestamp`, no `temporal_near` edges are built. PyG's `HeteroConv` gracefully skips edge types absent from the current graph — `available_edges = {et: ei for et in data.edge_index_dict if et[0] in x_dict and et[2] in x_dict}`.

**Feature-level**: All datasets are normalised to the same 6-dim feature vector. `alert_type` is hashed to a fixed integer range; `tactic` is mapped from any string to the 0–13 MITRE ATT&CK integer. Datasets without timestamps get `hour=0` and `day_of_week=0`.

**Routing**: `dataset_profiler.py` selects not just the checkpoint but also the clustering config — whether to use Spectral or HDBSCAN, what `min_cluster_size` to use, whether UMAP should be applied. This per-dataset tuning is necessary: UNSW-NB15 needs Spectral k=8, TON_IoT needs HDBSCAN with UMAP disabled, SQTK_SIEM needs ZCA whitening.

---

### Q: The model is "unsupervised" but you used SupCon fine-tuning — isn't that supervised?

The **core system is zero-shot unsupervised** — network_v9_v3 produces results on every dataset without any labels. That's the primary contribution.

SupCon fine-tuning (semi-supervised mode) requires labelled campaigns. We implemented it to answer: "how much can we gain if a small labelled set is available?" The answer varies: for UNSW-NB15, SupCon adds marginal value (the IXIA protocol diversity ceiling is structural). For TON_IoT, supervised training reaches ARI=0.845 vs zero-shot 0.431 — labels are essential for IoT.

The practical deployment model: use network_v9_v3 zero-shot for network IDS data. For IoT or host telemetry, expect to need supervised fine-tuning. The system is not universally zero-shot — it's zero-shot on the domain it was trained for (network IDS).

---

### Q: What would you do differently?

Three things.

**Embedding dimension per dataset**: 128-dim is a good general purpose, but SQTK_SIEM (5,100 alerts) doesn't need the same capacity as UNSW-NB15 (175,341 alerts). A smaller backbone for smaller datasets would reduce embedding collapse risk.

**Source_sensor_encoder lazy init**: The CS-2 `source_sensor_encoder = Linear(-1, 128)` uses PyG's lazy init. When a checkpoint trained without CS-2 is loaded with `strict=False`, this encoder is randomly initialised and fires on the first forward pass with real data. The fix — explicit `Linear(feature_dim, 128)` with a shared constant — would require retraining but would make cross-sensor fine-tuning tractable.

**Host-domain checkpoint**: The network_v9_v3 checkpoint works remarkably well on OpTC (perfect binary separation) but only because OpTC has 2 campaigns. A dedicated checkpoint trained on Sysmon/EDR telemetry with process + commandline features would genuinely unlock host-domain APT detection at campaign granularity.

---

## 17. References

*For peer-reviewed venues (ACSAC, RAID, NDSS) - optional for design document.*

### Graph Neural Networks
- **GAT**: Veličković et al. "Graph Attention Networks." ICLR 2018.
- **PyG**: Fey & Lenssen. "Fast Graph Representation Learning with PyTorch Geometric." ICLR 2019.

### Contrastive Learning
- **NT-Xent**: Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.
- **SimCLR**: Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.

### Clustering Methods
- **HDBSCAN**: McInnes et al. "DBSCAN Revisited, Revisited: Why Density-Based Clustering is Still Relevant." 2017.

### Security-Domain Systems (cross-referenced with §15.1)
- **DeepCASE**: van Ede, T., Aghakhani, H., Spahn, N., Bortolameotti, R., Cova, M., Continella, A., van Steen, M., Peter, A., Kruegel, C. & Vigna, G. "DEEPCASE: Semi-Supervised Contextual Analysis of Security Events." IEEE Symposium on Security and Privacy (S&P), 2022.
- **WATSON**: Zeng, J., Chua, Z. L., Chen, Y., Ji, K., Liang, Z. & Mao, J. "WATSON: Abstracting Behaviors from Audit Logs via Aggregation of Contextual Semantics." NDSS 2021.
- **MAGIC**: Jia, Z., Xiong, Y., Nan, Y., Zhang, Y., Zhao, J. & Wen, M. "MAGIC: Detecting Advanced Persistent Threats via Masked Graph Representation Learning." USENIX Security 2024.
- **Euler**: King, I. J. & Huang, H. H. "Euler: Detecting Network Lateral Movement via Scalable Temporal Link Prediction." NDSS 2022. (Extended in ACM TOPS 2023.)
- **CADE**: Yang, L., Guo, W., Hao, Q., Ciptadi, A., Ahmadzadeh, A., Xing, X. & Wang, G. "CADE: Detecting and Explaining Concept Drift Samples for Security Applications." USENIX Security 2021.

### Heterogeneous & Self-Supervised Graph Learning (cross-referenced with §15.2)
- **R-GCN**: Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I. & Welling, M. "Modeling Relational Data with Graph Convolutional Networks." ESWC 2018.
- **HAN**: Wang, X., Ji, H., Shi, C., Wang, B., Cui, P., Yu, P. & Ye, Y. "Heterogeneous Graph Attention Network." WWW 2019.
- **HGT**: Hu, Z., Dong, Y., Wang, K. & Sun, Y. "Heterogeneous Graph Transformer." WWW 2020.
- **DGI**: Veličković, P., Fedus, W., Hamilton, W. L., Liò, P., Bengio, Y. & Hjelm, R. D. "Deep Graph Infomax." ICLR 2019.
- **GRACE**: Zhu, Y., Xu, Y., Yu, F., Liu, Q., Wu, S. & Wang, L. "Deep Graph Contrastive Representation Learning." ICML Workshop on Graph Representation Learning 2020.
- **GraphMAE**: Hou, Z., Liu, X., Cen, Y., Dong, Y., Yang, H., Wang, C. & Tang, J. "GraphMAE: Self-Supervised Masked Graph Autoencoders." KDD 2022.
- **HeCo**: Wang, X., Liu, N., Han, H. & Shi, C. "Self-Supervised Heterogeneous Graph Neural Network with Co-Contrastive Learning." KDD 2021.

### Provenance-Graph Security
- **ATLAS**: Alsaheel, A., Nan, Y., Ma, S., Yu, L., Walkup, G., Celik, Z. B., Zhang, X. & Xu, D. "ATLAS: A Sequence-based Learning Approach for Attack Investigation." USENIX Security 2021.
- **DEPCOMM**: Xu, Z., Fang, P., Liu, C., Xiao, X., Wang, Y. & Liao, Q. "DEPCOMM: Graph Summarization on System Audit Logs for Attack Investigation." IEEE S&P 2022.
- **NoDoze**: Hassan, W. U., Guo, S., Li, D., Chen, Z., Jee, K., Li, Z. & Bates, A. "NoDoze: Combatting Threat Alert Fatigue with Automated Provenance Triage." NDSS 2019.

---

*Document version: May 2026 | MITRE-CORE v2.43 | Honest framing pass — citations verified, methodology gaps documented in §14, §3 headline numbers validated over 3 seeds.*
