# Report 3 (DEEP-DIVE): Self-Supervised / Zero-Shot Learning Claims
## Full Training Pipeline Audit

**Review Claim:** "MITRE-CORE uses self-supervised learning with topological NT-Xent contrastive loss for zero-shot campaign correlation across domains."

**Investigation Date:** 2026-03-05
**Method:** Traced every loss function → every training script → every checkpoint

---

## 1. Training Pipeline Map

| Checkpoint | Training Script | Loss Function(s) | Status |
|-----------|----------------|-------------------|--------|
| `multidomain_v2/best_supervised.pt` | `hgnn/hgnn_training.py` | `ContrastiveAlertLearner` (SimCLR) + `CrossEntropyLoss` | **CANONICAL** |
| `network_v9_v3/network_it_best.pt` | `train_graph_mae_v9_multidata_fast.py` | `topological_ntxent_loss()` + `CrossDomainContrastiveLoss` | **CANONICAL (zero-shot)** |
| `unsw_supcon_v7/best.pt` | `finetune_supcon.py` | `SupConLoss` (Khosla et al. 2020) | Semi-supervised |
| `train_hybrid_v10.py` output | `train_hybrid_v10.py` | `NTXentLoss` + `SupConLoss` + pseudo-label | **EXPERIMENTAL, FAILED** |
| `finetune_cross_sensor.py` output | `finetune_cross_sensor.py` | `NTXentLoss` | Cross-sensor only |

---

## 2. NT-Xent Usage Audit

### `NTXentLoss` class (`@hgnn/contrastive_loss.py:12-64`)

**Where used:**
- `train_hybrid_v10.py:38,88,151,240` — **EXPERIMENTAL, FAILED**. File header: "Caused UNSW-NB15 regression (<0.02 ARI). Not in canonical pipeline."
- `finetune_cross_sensor.py:52,298` — Cross-sensor finetuning only

**Where NOT used:**
- `train_graph_mae_v9_multidata_fast.py` — uses custom `topological_ntxent_loss()`, NOT the class
- `hgnn/hgnn_training.py` — uses `ContrastiveAlertLearner`, NOT `NTXentLoss`
- `finetune_supcon.py` — uses `SupConLoss`, NOT `NTXentLoss`

**Verdict:** `NTXentLoss` is **dead code** in the canonical pipeline. Only used in a script the authors marked as failed.

### `CrossGraphNTXentLoss` (`@hgnn/cross_domain_contrastive.py:18-105`)

Used only in `train_graph_mae_v9_multidata_fast.py:25,196` — and only when `--cross_graph` flag is passed (**off by default**: `use_cross_graph=False` at line 110).

---

## 3. What Actually Trained Each Checkpoint

### `multidomain_v2/best_supervised.pt`

**Phase 1 — `ContrastiveAlertLearner`** (`@hgnn/hgnn_correlation.py:2342-2349`):
```python
def forward(self, data1, data2):
    _, emb1 = self.hgnn(data1); _, emb2 = self.hgnn(data2)
    z1 = F.normalize(emb1["alert"], dim=1); z2 = F.normalize(emb2["alert"], dim=1)
    sim = torch.mm(z1, z2.t()) / self.temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return F.cross_entropy(sim, labels)
```
This is **SimCLR**, not NT-Xent. Instance-level contrastive: augmented views of same graph = positives.

**Phase 2 — Supervised Fine-tuning** (`@hgnn/hgnn_training.py:253-281`):
```python
unsw_criterion = nn.CrossEntropyLoss()
loss = unsw_criterion(cluster_logits, labels)  # ground-truth campaign IDs
```
Standard **supervised cross-entropy** on campaign labels. 10-class cluster classifier.

**The review's "self-supervised NT-Xent" claim is wrong on both counts: it's SimCLR, and Phase 2 is fully supervised.**

### `network_v9_v3/network_it_best.pt`

**Loss** (`@train_graph_mae_v9_multidata_fast.py:270`):
```python
loss = 1.0 * avg_topo + 0.5 * aug_ntxent
```
- `avg_topo` = `topological_ntxent_loss()` — custom InfoNCE using graph adjacency for positives
- `aug_ntxent` = `CrossDomainContrastiveLoss()` — SimCLR-style augmentation contrastive

This IS self-supervised. But the "topological" component is only 67% of the loss, and it uses co-occurrence edges (shared IPs/hostnames), not mathematical topology.

---

## 4. "Topological" Claim — Reality Check

The graph edges come from `AlertToGraphConverter`:
- Alerts sharing the same IP → edge
- Alerts sharing the same hostname → edge
- Alerts sharing the same username → edge

These are **co-occurrence relationships**, not topological features. No persistent homology, Betti numbers, or simplicial complexes are involved. The word "topological" is used to describe graph adjacency — a standard GNN input, not a novel contribution.

---

## 5. Zero-Shot Performance Reality

From `zeroshot_baseline_final.csv` (network_v9_v3 → all datasets):

| Dataset | ARI | AMI | Effective? |
|---------|-----|-----|-----------|
| NSL-KDD | 0.743 | 0.638 | YES |
| UNSW-NB15 | 0.538 | 0.664 | YES |
| SQTK_SIEM | 0.184 | 0.342 | MARGINAL |
| TON_IoT | 0.082 | 0.428 | NO |
| OpTC | 0.048 | 0.149 | NO |

Zero-shot works on **2/5 datasets**. On 3/5 it's near-random. The review implies generalizability the data does not support.

---

## 6. Revised Accuracy

| Sub-Claim | Before | After Deep-Dive |
|-----------|--------|-----------------|
| NT-Xent loss exists | 100% | 100% |
| NT-Xent is primary training method | 10% | **0%** |
| Zero-shot capability | 100% | 100% (but only 2/5 datasets) |
| "Topological" framing | 30% | **15%** |
| Pure unsupervised mode | 100% | 100% |
| Self-supervised learning | 50% | **35%** |

**Revised Overall: 35% (down from 55%)**

---

## 7. Critical Findings

1. **`NTXentLoss` is dead code.** Never used in any canonical training run. Only appears in a failed experimental script.

2. **`multidomain_v2` uses supervised labels.** Phase 2 trains with `CrossEntropyLoss` on ground-truth campaign IDs. Not self-supervised.

3. **"Topological" = co-occurrence edges.** No mathematical topology involved. Standard GNN adjacency.

4. **Zero-shot is unreliable.** Fails on 3/5 datasets. The review presents it as a general capability.
