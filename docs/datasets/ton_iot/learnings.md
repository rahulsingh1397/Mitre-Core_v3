# TON_IoT — Learnings

Filled at Stage 6 freeze — 2026-05-20.
Carry these forward to CICIDS2017 and subsequent datasets.

---

## L1 — HDBSCAN mcs must be calibrated to the dataset's class-size regime

The NSL-KDD default `hdbscan_min_cluster_size=5` is calibrated for datasets where each class has ~100–500 rows in a 10K subset. On TON_IoT, each of the 8 main attack classes has ~950–1,000 rows. At mcs=5, HDBSCAN splits each large class into 5–6 sub-clusters, yielding 55 predicted clusters for 10 true classes. The fix (mcs=300) improved eval ARI from 0.423 → 0.474 but still could not close the gap to K-Means(raw). **Carry-forward:** when a new dataset has balanced large classes, set mcs ≈ 10–30% of the smallest expected class size in the eval subset as a starting grid point.

## L2 — K-Means has a privileged prior on balanced datasets

K-Means receives `n_clusters=10` as input — exactly the true class count on TON_IoT. This is an unfair advantage in any deployment scenario where the number of campaigns is unknown. Any paper comparison must note this. V3 with HDBSCAN makes a truly unsupervised cluster-count estimate (17 at mcs=300). **Carry-forward:** when reporting method rankings, annotate K-Means results with "(n_clusters = true class count)" to flag the privileged prior.

## L3 — AMI and ARI diverge when HDBSCAN over-segments

V3 ARI=0.423 vs AMI=0.705 on TON_IoT (and ARI=0.474 vs AMI=0.702 after sweep). The ~0.28 gap is the diagnostic signature of over-segmentation: AMI is less sensitive to splitting a true cluster into many small sub-clusters than ARI is. When ARI is low but AMI is competitive, the HGNN representation is good — the clustering step is the bottleneck, not the embedding. **Carry-forward:** always report both metrics and inspect the ARI/AMI gap as a diagnostic for over- vs under-segmentation.

## L4 — IP-only graph degrades gracefully; missing hostname/username is not catastrophic

TON_IoT lacks `hostname` and `username` columns (present in NSL-KDD/UNSW). `AlertToGraphConverter` handles this via `dropna()` fallback — it simply constructs fewer entity node types (IP-only rather than IP+host+user). The smoke test (29 clusters, 0% noise on 500 rows) and the competitive AMI=0.705 both confirm the backbone remains useful. **Carry-forward:** check for hostname/username in the audit phase but do not treat their absence as a blocker; document the graph reduction and move on.

## L5 — dominant_confusion_accuracy is structurally degenerate on well-separated multi-class data

This is the third consecutive dataset (NSL-KDD, UNSW-NB15, TON_IoT) where `dominant_confusion_accuracy` is constant 1.0 across all methods. It is not dataset-specific noise — it is a metric design flaw on well-separated multi-class benchmarks. **Carry-forward:** demote this metric globally in future datasets without re-running the diagnostic. Update the metric demotion note in `docs/datasets/_template/` to pre-demote it.

## L6 — Path B (V3 loses) is a publishable finding, not a lifecycle failure

TON_IoT revealed genuine limitations: zero-shot transfer from network IDS to IoT traffic is imperfect, and the HDBSCAN clustering step needs per-dataset calibration. This is scientifically valuable. The lifecycle worked exactly as designed — the investigation + honest cap + frozen result produces a credible negative result that strengthens the paper (it shows the benchmark is not gamed). **Carry-forward:** Path B outcomes are first-class results; document them with the same rigor as Path A wins.
