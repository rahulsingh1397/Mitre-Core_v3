# CICIDS2017 — Decision Log

Format: `## YYYY-MM-DD — <Choice made>`

---

## 2026-05-20 — n_clusters=15 (full schema count)

**Choice:** Set `n_clusters=15` for K-Means baselines, matching the full corpus alert_type class count.
**Alternative considered:** n_clusters=12 (evaluable classes in 10K only), n_clusters=10 (match other datasets).
**Rationale:** K-Means's privileged prior should reflect the true class count. The 3 absent classes in 10K result in empty clusters — known behavior, documented in audit.md. Full schema count is consistent with the principle used for NSL-KDD (10 classes → n_clusters=10).

## 2026-05-20 — Null alert_type rows treated as UNKNOWN (not excluded)

**Choice:** Accept the benchmark's `fillna("UNKNOWN")` treatment of 288,602 null alert_type rows.
**Alternative considered:** Patching the benchmark loader to exclude null-label rows before stratified sampling.
**Rationale:** The benchmark code already handles this correctly — null rows become "UNKNOWN" class and are sampled proportionally (~925 rows in 10K). In the graph they are isolated nodes (null src_ip/dst_ip). Patching the loader would introduce a CICIDS2017-specific code path deviating from the lifecycle established for prior datasets. The metric penalty from UNKNOWN rows is real and is documented in audit.md.

## 2026-05-20 — Sweep winner: mcs=200, pca=8, eps=0.15 (freeze config)

**Choice:** Use sweep winner engine_kwargs for the frozen baseline (not default config).
**Alternatives considered:** (a) default config for cross-dataset consistency like TON_IoT; (b) eps=0.15 only with default mcs=5 (nearly identical ARI=0.176 vs 0.177).
**Rationale:** Default config produces 45 clusters for 15 true classes (3× over-segmented). Sweep winner produces 11 clusters — geometrically correct. The improvement is real (+60% relative ARI). Unlike TON_IoT (default ARI=0.423, reasonable geometry), CICIDS2017 default is poorly calibrated for 72.9% BENIGN class. V3 still loses to Spectral(emb) at 0.333 — this is a confirmed finding regardless of config.
**Sweep artifact:** `benchmark/results/latest/cicids2017/sweep_full_engine.csv`

## 2026-05-20 — Stage 4 Path B: V3 loses to Spectral(emb) by 0.222 ARI

**Choice:** Proceed with full-engine sweep (Path B).
**Evidence:** V3 ARI=0.111 ± 0.000 (45 clusters), Spectral(emb) ARI=0.333 ± 0.034 (15 clusters). Margin = -0.222.
**Root cause identified:** BENIGN=72.9% in 10K (7,290 rows); HDBSCAN mcs=5 over-segments BENIGN into ~30 sub-clusters. 45 total clusters vs 15 true classes.
**Sweep target:** expanded mcs grid [5,50,100,200,300,500,1000] on dev split.
**Note on DBSCAN:** DBSCAN nominally ARI=0.395 but n_clusters=2, noise_fraction=0.908 → degenerate binary split, attack_f1_demoted=0.000. Not a valid baseline winner.

## 2026-05-20 — v1.0 Freeze

**Choice:** Freeze CICIDS2017 baseline at sweep winner config (mcs=200, pca=8, eps=0.15).
**Frozen metrics (eval split, label_col track, seeds 42/43/44):**
- V3 ARI = 0.1771 ± 0.000 (rank 2nd; perfectly deterministic)
- V3 AMI = 0.5699 ± 0.000 (rank 1st; highest of all methods)
- Spectral (emb) ARI = 0.333 ± 0.028 (best non-degenerate baseline; rank 1st on ARI)
- DBSCAN demoted: n_clusters=2, noise_fraction=0.908, attack_f1_demoted=0.000
- dominant_confusion_accuracy: 1.0 for all methods (pre-demoted, 4th consecutive dataset)
**Artifacts:** `benchmark/results/frozen/cicids2017/v1.0/` (11 files)
**Tag:** `cicids2017-v1.0`

## 2026-05-20 — Same engine_kwargs as TON_IoT (no Stage 2 retune)

**Choice:** Use the same engine configuration as NSL-KDD/UNSW/TON_IoT at Stage 2.
**Alternative considered:** Pre-tuning hdbscan_min_cluster_size for CICIDS2017's imbalance profile.
**Rationale:** Per lifecycle protocol, per-dataset retuning only happens in Stage 4 Path B (if V3 loses). Running Stage 2 with default config ensures the Stage 3 baseline roster reflects the out-of-the-box V3 performance honestly.
