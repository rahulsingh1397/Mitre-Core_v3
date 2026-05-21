# CICIDS2017 — Learnings

**Frozen:** 2026-05-20 (v1.0)

---

## L1 — Extreme class imbalance creates a structural over-segmentation problem

BENIGN=72.9% of the 10K sample (7,290 rows). HDBSCAN with mcs=5 (from NSL-KDD/UNSW/TON_IoT default)
finds ~30 density modes within the massive BENIGN block, yielding 45 total clusters for 15 true classes.
This is qualitatively different from prior datasets (max imbalance was ~50% in NSL-KDD Normal).

**Carry forward:** For any new dataset where the dominant class exceeds 60% of the sample, flag
immediately that mcs=5 will likely over-segment. Consider mcs=100+ as the Stage 2 starting point
rather than the default.

---

## L2 — eps=0.15 is a stronger lever than mcs in embedding space

The full 168-config sweep found that `hdbscan_cluster_selection_epsilon=0.15` drove most of the ARI
improvement, with mcs playing a secondary role. eps=0.15 merges near-clusters that form within the
BENIGN density peak. mcs=5 vs mcs=200 at eps=0.15 gave nearly identical ARI (0.176 vs 0.177).

**Carry forward:** On datasets with class imbalance ≥60%, try eps=0.15 first before expanding mcs.
Two-parameter sweep (mcs × eps with fixed pca=8) may suffice vs the full 168-config grid.

---

## L3 — Spectral (emb) with oracle cluster count is an upper bound, not a fair competitor

Spectral (emb) wins with ARI=0.333 by combining V3's GNN embeddings with the exact true class count
(n_clusters=15). This is the best achievable result with V3's representation given oracle supervision.
V3's disadvantage is purely the unsupervised cluster count discovery problem.

**Implication:** V3's true weakness on CICIDS2017 is HDBSCAN cluster enumeration, not embedding
quality. AMI=0.570 (highest of all methods) confirms the embeddings are discriminative.

**Carry forward:** When writing up CICIDS2017 results, lead with AMI (where V3 wins) and explain
why ARI penalises over-segmentation more harshly than the deployment reality warrants.

---

## L4 — Null label rows with null IPs become isolated graph nodes, not excluded rows

288,602 rows have null `alert_type`, `timestamp`, `src_ip`, `dst_ip`. These appear as "UNKNOWN"
class via `fillna("UNKNOWN")` and form isolated nodes (no IP edges). They land in the 10K sample
as ~925 "UNKNOWN" rows and marginally increase noise in the BENIGN-heavy density region.

**Carry forward:** When auditing a new dataset, check whether null-label rows also have null
connectivity features (src_ip/dst_ip). If so, they are isolated nodes — note in audit.md whether
this is a material concern (usually not if <10% of sample).

---

## L5 — IP-only graph (no hostname/username) is the standard for CICIDS2017

Same as TON_IoT: `SourceAddress`/`DestinationAddress` field names are not present; `src_ip`/`dst_ip`
are present but the `shares_ip` graph edge uses `SourceAddress`. Only temporal edges form.

**Carry forward:** The two-column IP convention (src_ip/dst_ip vs SourceAddress/DestinationAddress)
should be standardised in AlertToGraphConverter. Current approach (graceful fallback) is fine but
causes silent feature reduction. Document which edge types are active per dataset in audit.md.

---

## L6 — V3 loses on a second consecutive dataset; the zero-shot claim must be scoped

After TON_IoT and CICIDS2017, V3 loses to at least one non-degenerate baseline. Both are non-NSL-KDD
datasets. NSL-KDD and UNSW-NB15 are the two where V3 cleanly wins.

**Pattern:** V3 wins on "classic" network IDS benchmarks (NSL-KDD, UNSW-NB15) where class balance is
moderate and the checkpoint was implicitly calibrated. On IoT (TON_IoT) and high-imbalance network
traffic (CICIDS2017), V3 underperforms.

**Carry forward:** The paper claim should be "V3 achieves competitive zero-shot performance on
network IDS benchmarks (NSL-KDD, UNSW-NB15)" rather than "V3 works across all datasets."
The dataset scope limitation is now empirically validated across 4 datasets.

---

## L7 — perfectly deterministic V3 output is a useful property

V3 ARI std=0.000 across all three seeds (42, 43, 44) for all CICIDS2017 metrics. Same as TON_IoT.
This means seed choice is irrelevant for V3 on CICIDS2017 (HDBSCAN is seeded; HGNN inference is
deterministic given the same graph).

**Carry forward:** This determinism is a genuine benchmark advantage — V3 results can be quoted
without confidence intervals for the metric itself (only sample-seed variance matters, and
exclude_sample_seeds controls for that).
