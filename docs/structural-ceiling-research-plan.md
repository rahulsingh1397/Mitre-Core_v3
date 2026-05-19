# Breaking the MITRE-CORE Structural Ceiling: Deep Research Plan

A multi-track research program to break the 0.17/0.40/0.50 ARI ceilings on SQTK, UNSW-NB15, and CICIDS2017 via prototype learning, richer features, architectural upgrades, and a proper `alt_ari` redesign.

---

## 1. Problem Diagnosis (ceilings are not independent)

| Dataset | Current | Target | Root cause (hypothesis) |
|---|---|---|---|
| **UNSW-NB15** | 0.367 (v7 SupCon) | >0.40 | Representation plateau; 3 tiny campaigns unseparable at `tau=0.07`, single-view; backbone pretrained on `network_v9_v3` not UNSW-native |
| **CICIDS2017** | HGNN AMI=0.500 vs GMM=0.630 | >0.63 | Architectural: 1-layer HGNN + mean aggregation loses the continuous flow-statistic structure that GMM captures; alert-alert edges too sparse post gate |
| **SQTK_SIEM** | 0.174 | >0.25 | Graph topology ceiling: few cross-alert bridges, most alerts isolated -> HDBSCAN treats majority as noise |
| **`alt_ari`** | meaningless | useful | `campaign_id = hash(attack_cat) % 50` in `training/download_datasets.py:293` - literally the same partition as `attack_cat` |

Key code anchors verified:
- `hgnn/hgnn_correlation.py:155-256` - `MITREHeteroGNN` (1 layer default, GATConv, mean HeteroConv aggr, LayerNorm+residual)
- `training/download_datasets.py:289-296` - `campaign_id` derived from `attack_cat`
- `training/finetune_supcon.py:263-314` - SupCon hyperparameters

---

## 2. Four Research Tracks

### Track A - Prototype Learning (Fix 4)
**Hypothesis:** Replacing implicit cluster structure with explicit learnable class prototypes removes the dependence on HDBSCAN's density assumption and should help *all three* datasets, especially SQTK where density-based clustering fails.

**Methods to evaluate:**
1. **ProtoNCE / SwAV-style prototypes** - `K` learnable vectors `C in R^{K×d}`; Sinkhorn-Knopp equipartition on embeddings; cross-entropy to soft cluster codes. No ground-truth labels needed.
2. **Supervised prototype head** - Weighted class centroids from SupCon embeddings; inference uses cosine-NN to prototypes (replaces HDBSCAN entirely).
3. **DeepCluster-v2 on HGNN** - Iterative k-means -> pseudo-labels -> cross-entropy, with feature whitening (we already have DeepCluster baseline code).

**Expected gains:** UNSW +0.05-0.10 ARI (separates tiny campaigns via explicit centroids); SQTK +0.05-0.08 (no density threshold); CICIDS +0.03 (modest - not the bottleneck).

**Implementation cost:** Medium. New loss head in `hgnn_correlation.py`; new trainer variant `training/train_protonce.py`. ~2-3 days.

**Risk:** Requires knowing K. Mitigate with Sinkhorn-Knopp + overclustering (K=3×true_classes).

---

### Track B - Better Features (highest impact for CICIDS + SQTK)
**Hypothesis:** CICIDS gap vs GMM proves HGNN is *discarding* information present in raw flow statistics. SQTK ceiling is a feature-sparsity problem.

**Concrete additions (in priority order):**

| # | Feature | Target dataset | Mechanism |
|---|---|---|---|
| B1 | **Temporal burstiness** (rolling 1s/10s/60s alert counts per src_ip) | All | 3 new continuous features in `AlertToGraphConverter` |
| B2 | **Command-sequence n-grams (3/5-gram hashed)** | OpTC, SQTK | MinHash signature -> 32-dim sketch fed into `command_line_encoder` |
| B3 | **Flow-statistic tail features** (p95/p99 of sbytes/dbytes per 5-alert window) | CICIDS, UNSW | Capture the heavy-tailed distributions GMM models naturally |
| B4 | **External threat intel** (IP reputation via MaxMind + AlienVault OTX lookup, cached) | All (esp. SQTK) | New `threat_intel` node type with 16-dim encoding |
| B5 | **Graph motif features** (triangle count, degree centrality per alert) | SQTK, CICIDS | Pre-computed via NetworkX; adds structural signal to isolated alerts |

**Expected gains:** CICIDS +0.10-0.15 AMI (B3 closes GMM gap); SQTK +0.05-0.10 (B4+B5 adds bridges to isolated alerts); UNSW +0.03 (marginal).

**Cost:** B1/B3/B5 are cheap (~1 day each). B2 medium. B4 has external-API and caching complexity (~3 days).

---

### Track C - Architectural Upgrades
**Hypothesis:** 1-layer mean-aggregated HeteroConv under-smooths complex graphs (CICIDS) while 2+ layers over-smooth (confirmed in prior memos). Solution: attention-based aggregation with gating.

**Candidates:**

1. **HGT (Heterogeneous Graph Transformer)** - Replaces `HeteroConv({rel: GATConv})` with per-relation Transformer attention + edge-type-aware keys/queries. Paper: Hu et al. 2020.
2. **HAN (Heterogeneous Attention Network)** - Meta-path-level attention on top of GAT; explicitly weights `shares_ip` vs `shares_host` vs `temporal_near`.
3. **GraphGPS** - Message passing + global Performer attention. Best for SQTK (captures long-range dependencies bypassing sparse graph).
4. **PNA (Principal Neighbourhood Aggregation)** - Replace `aggr="mean"` with `[mean, max, std, sum]` multi-aggregation. Cheapest fix - single file change.
5. **JK-Net / GCNII** - Jumping knowledge + identity mapping; prevents over-smoothing at 3+ layers.

**Recommended ablation order (cheap->expensive):**
PNA aggregation -> JK-Net (2 layers) -> HAN meta-path attention -> HGT -> GraphGPS.

**Expected gains:** CICIDS +0.08 (PNA captures variance GMM uses); UNSW +0.04; SQTK +0.06 (HGT/GraphGPS bypass sparsity).

**Cost:** PNA ~2 hours. HGT/GraphGPS ~1 week each (PyG has reference implementations).

---

### Track D - Fix `alt_ari` (make evaluation honest)
**Problem:** `campaign_id = hash(attack_cat) % 50` in `training/download_datasets.py:293` collapses `alt_ari` onto `ari`.

**Proposed independent partitions (pick 2-3):**
1. **Temporal-session partition** - Group alerts into sessions via 30-min inactivity gap per `src_ip`; `alt_campaign_id = session_id`. Independent of attack_cat.
2. **Connected-component partition** - Build graph on `(src_ip, dst_ip)` edges; CC labels become `alt_campaign_id`. Measures whether HGNN recovers IP-based cohorts without labels.
3. **Tactic-chain partition** - MITRE tactic sequences grouped by `src_ip` over 5-alert windows; labels = distinct tactic n-grams. Measures if embeddings capture attacker behavior rather than category.

**Deliverable:** New column `alt_campaign_id_v2` added by `scripts/add_alt_labels.py`; update gate-tuning/baseline CSVs to report `ari` (attack_cat), `alt_ari_temporal`, `alt_ari_graph`.

**Cost:** Low (~1 day). High research value (reviewers will ask about label leakage).

---

## 3. Experimental Matrix

| Experiment | Track | Dataset(s) | Baseline ARI | Target | Estimated effort |
|---|---|---|---|---|---|
| E1. PNA aggregation ablation | C | all | current | +0.03-0.08 | 0.5 day |
| E2. Temporal burstiness features (B1) | B | all | current | +0.02-0.05 | 1 day |
| E3. Flow-tail features (B3) | B | CICIDS, UNSW | 0.500 AMI | 0.60+ | 1 day |
| E4. ProtoNCE head | A | all | current | +0.05-0.10 | 2-3 days |
| E5. Supervised prototype inference | A | UNSW | 0.367 | 0.42+ | 1 day |
| E6. HAN meta-path attention | C | SQTK, CICIDS | 0.174 / 0.500 | 0.22 / 0.58 | 3 days |
| E7. Independent `alt_ari` labels | D | all | n/a | honest metric | 1 day |
| E8. Graph motif features (B5) | B | SQTK | 0.174 | 0.22+ | 1 day |
| E9. Threat-intel node (B4) | B | SQTK | 0.174 | 0.25+ | 3 days |
| E10. HGT end-to-end | C | all | current | stretch goal | 1 week |

**Total minimum viable path (E1+E2+E3+E5+E7):** ~5 days, targets all three ceilings.

---

## 4. Recommended Sequencing (8-day plan)

1. **Day 1** - E7 (`alt_ari` fix) + E1 (PNA). Cheap wins, unblocks honest measurement.
2. **Day 2** - E2 + E3 (burstiness + tail features). Expected to close CICIDS gap.
3. **Day 3** - E5 (prototype inference on UNSW). Target 0.40+.
4. **Day 4-5** - E4 (ProtoNCE training across all datasets). Primary contribution.
5. **Day 6** - E8 (graph motifs for SQTK).
6. **Day 7-8** - E6 (HAN) if SQTK still below 0.25.

Defer E9 (threat intel) and E10 (HGT) unless time permits - external dependencies / large engineering.

---

## 5. Success Criteria

- UNSW-NB15 ARI > 0.40 on test split
- CICIDS2017 AMI > 0.60 (match or beat raw GMM)
- SQTK_SIEM ARI > 0.25
- `alt_ari` reports 2+ independent partitions with documented derivation
- All experiments logged to `experiments/results/ceiling_fix_v1.csv` with seed-averaged (n=3) statistics + 95% CI

---

## 6. Open Questions for User

1. **External threat intel (B4)** - is adding a MaxMind/OTX dependency acceptable, or should we keep the system self-contained?
2. **ProtoNCE vs DeepCluster-v2** - preference? (ProtoNCE is simpler; DeepCluster has existing code.)
3. **Should `alt_ari` partitions replace the current meaningless one in all CSVs, or be added as new columns?**
4. **Time budget** - is the 8-day sequencing acceptable, or should we prioritize a subset for a near-term submission?
