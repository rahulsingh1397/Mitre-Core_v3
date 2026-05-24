# Master Plan v1.1

> **Diff from v1.0:** Adds Part X (V3 Improvement Plan) as authorised execution path.
> v1.0 prescribed Part IV.8 (heavy GNN baselines) as next; v1.1 supersedes that with
> Part X experiments. Specific changes:
> - Part X.1 Exp 1 (hard neg mining) — executed, REJECTED 2026-05-22
> - Part X.2 Exp 2 (collapse diagnostic) — executed, hypothesis REJECTED 2026-05-23
> - Part X.2.D documentation correction batch — applied 2026-05-23 (5 SQTK_SIEM docs corrected)
> - Part X.2.5 Exp 2.5 (GMM+BIC clustering) — TON_IoT pass +0.181 ARI, CICIDS2017 fail
> - Part X.2.6 Exp 2.6 (SQTK_SIEM PCA sweep) — queued
> - v1.1 frozen artifact policy: any v1.1 freeze produces a new directory alongside the unchanged v1.0
>
> v1.0 frozen artifacts remain immutable. v1.0 baseline docs receive correction notes
> when measurement bugs are discovered (e.g. SQTK_SIEM cosine_sim).
>
> See `docs/plans/MASTER_PLAN_v1.0.md` for the original frozen plan.
> See `docs/experiments/revised_failure_map.md` for the updated experiment lineup.

---

# MITRE-CORE V3 — Master Plan: Freeze, Cleanup, Dataset-by-Dataset Improvement

## Context

NSL-KDD has been brought to a stable, validated state through Phases 0–5: protocol hardened, three label tracks, ten baselines, twelve metrics, dev/eval split frozen, MITRE-CORE V3 holding a ~0.19 ARI margin over the strongest baseline (Spectral on raw features). The remaining work is no longer about NSL-KDD — it's about (a) locking that result so it cannot drift, (b) cleaning V2 baggage that didn't contribute, and (c) repeating the same hardened lifecycle on the remaining datasets in a documented, auditable way.

This plan establishes:
1. An immutable **NSL-KDD v1.0 frozen baseline** as a permanent reference point.
2. A **codebase cleanup pass** that removes only what is provably unused by the validated V3 path.
3. A reusable **per-dataset improvement lifecycle** with documentation discipline.
4. An ordered roadmap across all benchmark datasets, each becoming its own versioned, frozen checkpoint when complete.

Nothing in Part I or II changes model behavior. Part III onward extends the benchmark; it does not retune NSL-KDD.

---

## Part I — Freeze NSL-KDD as v1.0 Baseline

**Goal:** make the current NSL-KDD numbers permanently reproducible and immune to silent drift from later refactors.

### I.1 Canonicalize artifacts
Move the Phase 5 result set into an immutable folder:
- `benchmark/results/frozen/nsl_kdd/v1.0/`
  - `results.csv` ← from `benchmark_real_results_phase5.csv`
  - `summary.csv` ← from `benchmark_real_results_phase5_summary.csv`
  - `manifest.json` ← from `benchmark_real_results_phase5_manifest.json`
  - `splits_dev_seed42.npy`, `splits_dev_seed42.json` (copied)
  - `splits_eval_seed142.npy`, `splits_eval_seed142.json` (copied)
  - `engine_config.yaml` — captured `engine_kwargs` at freeze time
  - `checkpoint_sha256.txt` — hash of `hgnn_checkpoints/network_v9_v3/network_it_best.pt`
  - `dataset_sha256.txt` — hash of `datasets/nsl_kdd/mitre_format.csv`
  - `environment.txt` — `pip freeze` output

This directory is read-only by convention; later phases never overwrite it.

### I.2 Write the canonical baseline doc
Create `docs/datasets/nsl_kdd/v1.0_baseline.md` containing:
- Headline metrics table (all 10 methods × tactic / alert_type / campaign_id tracks)
- Frozen protocol (split sizes, seeds, sample seeds, exclude_sample_seeds)
- Engine configuration locked at freeze time
- Hashes (dataset, checkpoint, splits)
- Reproduction command (single line)
- Known limitations (attack_f1 saturation, Phase 3 sweep methodology debt, single-checkpoint dependence)
- Pointer to `IMPROVEMENT_LOG.md` lines 43–261 for the full execution history

### I.3 Re-runnability test
New `tests/test_nsl_kdd_v1_frozen.py`:
- Loads `benchmark/results/frozen/nsl_kdd/v1.0/manifest.json`
- Verifies dataset + checkpoint hashes match on-disk files
- Runs the V3 lane on the eval subset only
- Asserts ARI/AMI match the frozen values to within 1e-6 (deterministic path) or 1e-3 if any nondeterminism is unavoidable
- Skipped in smoke CI, run in nightly / on-demand

### I.4 Git tag
Tag the freeze commit `nsl-kdd-v1.0` so future bisects can locate the baseline state exactly.

**Verification:** `pytest tests/test_nsl_kdd_v1_frozen.py` reproduces the frozen numbers; `docs/datasets/nsl_kdd/v1.0_baseline.md` answers "what were the numbers and how were they produced" without needing any other file.

---

## Part II — Codebase Cleanup (Quarantine-First)

**Goal:** remove what didn't contribute to the validated NSL-KDD result, without breaking anything that did.

**Strategy:** *quarantine before deletion.* Each batch is moved to `archive/v2_legacy/<batch>/`, then `pytest -q` runs on the full test suite. If green, mark the batch deletable; if red, restore exactly what's needed and re-quarantine the rest.

### II.1 Dependency map (build first, delete nothing yet)
Run a one-time import scan:
- `python -c "import mitre_core, benchmark, tests"` plus a `grep`-based import graph rooted at `mitre_core/`, `benchmark/`, `tests/`, `mitre_core_run.py`, `training/train_graph_mae_v9_multidata_fast.py`.
- Produce `docs/cleanup/dependency_map_v1.md` listing files reachable from the validated V3 entry points.
- Anything not reachable is a cleanup candidate. Anything reachable stays.

### II.2 Batches (in order; each is its own commit)

| Batch | Quarantine target | Why | Risk |
|---|---|---|---|
| **B1 — App stack** | `static/`, `templates/`, `stakeholder_dashboard.py`, `prepare_siem_data.py`, `siem/`, `ingestion/`, `core/` | V2 Flask/RL/SIEM app layer. Not used by `mitre_core/`, `benchmark/`, or tests. | Low |
| **B2 — Top-level dupes** | `evaluation/` (empty stub), `models/` (replaced by `mitre_core/models/`), `baselines/` (replaced by `benchmark/methods/`) | Pure duplicates of the V3 package surface. | Low |
| **B3 — Agentic** | `agentic/` | Not on the V3 benchmark path. | Low |
| **B4 — Dead utils** | `utils/rl_*.py`, `utils/cross_*.py`, `utils/long_range_temporal.py`, `utils/temporal_fragment_merger.py`, `utils/union_find.py`, `utils/semantic_encoder.py`, `utils/mitre_complete.py`, `utils/scalable_clustering.py`, `utils/explainability.py`, `utils/analyst_feedback.py` | Verify each is import-unreachable from V3, then quarantine. | Medium (run dep scan twice) |
| **B5 — Stale hgnn/** | `hgnn/cross_domain_contrastive.py`, `hgnn/domain_adaptation.py`, `hgnn/graph_mae.py`, `hgnn/hgnn_explainability.py`, `hgnn/hgnn_integration.py`, `hgnn/temporal_enrichment.py`, `hgnn/vgae_pretraining.py` | V2 experimental paths. Keep `hgnn_correlation.py`, `categorical_encoder.py`, `hgnn_evaluation.py`, `hgnn_training.py`, `models/`. | Medium |
| **B6 — Stale training** | `training/adapt_to_domain.py`, `training/finetune_supcon.py`, `training/train_link_pred.py`, `training/train_prototypes.py`, `training/train_vgae.py`, `training/download_datasets.py`, `training/large_dataset_downloader.py`, `training/modern_loader.py`, `training/train_on_datasets.py` | Non-V3 / supervised / prototype training. Keep `train_graph_mae_v9_multidata_fast.py`, `attack_data_loaders.py`, `training_base.py`. | Medium |
| **B7 — Experiments** | Move all `experiments/*.py` except those wired into `benchmark/` to `archive/v2_legacy/experiments/`. Keep `experiments/sweeps/` (new home for future per-dataset sweeps). | Many one-off V2 scripts; the validated V3 sweep apparatus lives in `benchmark/clustering_sweep.py`. | Medium |
| **B8 — Result artifact consolidation** | Move `benchmark/results/benchmark_real_results.csv`, `_frozen.csv`, `_phase3.csv`, `_phase4.csv`, `e2e_benchmark_results.csv`, `nsl_kdd_clustering_sweep.csv` into `benchmark/results/history/nsl_kdd/`. Keep `benchmark/results/frozen/nsl_kdd/v1.0/` (canonical) and `benchmark/results/latest/` (working dir). | Reduce ambiguity about which CSV is "the answer." | Low |
| **B9 — Docs archive** | Move `docs/2403.09118v1.pdf`, `docs/GenAI-Red-Teaming-Guide-*.pdf`, `docs/IEEE_TIFS_*.docx`, `docs/Learning_the_Associations_*.pdf`, `docs/electronics-*.pdf`, `docs/PROJECT_SUMMARY_UPDATED.pdf`, `docs/MITRE_CORE_Upgrade_Plan.docx`, `docs/TABLE_VIII_*.md`, `docs/WEEK4_V10_*.md`, `docs/v3_ari_test_results.md`, `docs/v3_finetuning_results.md` → `docs/archive/v2/`. | Keep `architecture.md`, `benchmark_protocol.md`, `reproducibility.md`, `competitive_analysis.md`, `structural-ceiling-research-plan.md` in the active docs root. | Low |

### II.3 Address the Phase 3 methodology debt
- Rename `benchmark/clustering_sweep.py` → `benchmark/clustering_sweep_standalone.py`.
- Add a docstring banner: *"WARNING: standalone HDBSCAN sweeps do not transfer to the full V3 pipeline. Phase 3 (NSL-KDD) demonstrated catastrophic ARI degradation when winners were ported. Use `clustering_sweep_full_engine.py` instead."*
- Create `benchmark/clustering_sweep_full_engine.py` skeleton: same grid interface, but routes every config through `V3CorrelationEngine.cluster_alerts(...)` so winners actually transfer.
- Document the lesson in `docs/lessons/phase3_sweep_methodology.md`.

### II.4 Cleanup validation gate
After each batch:
1. `pytest -q` — all tests pass.
2. `python -m benchmark.run_benchmark --config benchmark/datasets_real.yaml --eval-only --output benchmark/results/latest/cleanup_smoke.csv` — NSL-KDD V3 still produces the frozen ARI/AMI.
3. Commit with message `cleanup(Bn): quarantine <category>`.

After all batches green: a single follow-up commit deletes `archive/v2_legacy/` contents that have stayed untouched for the cleanup pass. Anything still in doubt stays archived, not deleted.

**Verification:** repo `tree -L 2` post-cleanup shows a focused V3 layout matching the spec in CLAUDE.md / `MITRE-CORE V3.md`. NSL-KDD v1.0 frozen test still passes.

---

## Part III — Per-Dataset Improvement Lifecycle (the reusable pattern)

**Goal:** every remaining dataset follows the same documented six-phase lifecycle that worked for NSL-KDD. No bespoke flows.

### III.1 Standard six-phase lifecycle (per dataset)
1. **Audit** — class counts, label semantics, candidate label columns, data hash, row count, schema sanity.
2. **Protocol freeze** — dev + eval sample seeds chosen, indices persisted to `benchmark/splits/`, manifest emitted.
3. **Baseline roster** — all 10 baselines (raw + PCA + embedding-based + V3) run on the eval subset.
4. **Full-engine sweep** (only if needed) — clustering hyperparameters swept through `clustering_sweep_full_engine.py`. Skip if V3 already dominates baselines by >0.1 ARI.
5. **Metrics + demotion** — full 12-metric set, attack_f1_demoted, cluster diagnostics.
6. **Freeze v1.0** — `benchmark/results/frozen/<dataset>/v1.0/` written, doc published, git-tagged.

### III.2 Per-dataset documentation template
Each dataset gets a folder `docs/datasets/<name>/` containing:
- `README.md` — current status (in-progress / frozen / blocked), pointer to canonical baseline version
- `v1.0_baseline.md` — once frozen (template mirrors NSL-KDD's)
- `audit.md` — Phase 1 audit findings
- `protocol.md` — split + seed decisions, with rationale
- `decision_log.md` — every non-trivial choice, dated, with the alternative considered
- `learnings.md` — what surprised us, what didn't transfer, what to carry to the next dataset

The repo-level `IMPROVEMENT_LOG.md` becomes an index: chronological one-line entries pointing at the per-dataset docs.

### III.3 Standard per-dataset files
- `benchmark/datasets_real.yaml` — one entry block per dataset (dev + eval pair when applicable)
- `datasets/loaders/<name>.py` — surfaces all label tracks from a single CSV pass
- `benchmark/splits/<name>_<size>_seed<seed>.{npy,json}` — frozen indices
- `benchmark/results/frozen/<name>/v1.0/` — immutable
- `benchmark/results/latest/<name>/` — working area
- `benchmark/results/history/<name>/` — superseded artifacts

### III.4 Reusable tests
- `tests/test_<dataset>_v1_frozen.py` — same pattern as NSL-KDD's
- `tests/test_split_disjoint.py` — generalized to assert disjointness for every dataset with dev+eval seeds in YAML

---

## Part IV — Dataset Order & Per-Dataset Subplans

Order chosen to maximize learning per unit of effort: closest-to-NSL-KDD first (proves the pattern transfers), then progressively further from the NSL-KDD assumptions.

### IV.1 NSL-KDD — **v1.0 FROZEN** (Part I)
Reference baseline. No further work unless a future model change requires re-evaluation, in which case it becomes v1.1 in a new folder; v1.0 stays immutable.

### IV.2 UNSW-NB15 — **next active dataset**
Closest analog to NSL-KDD. Network IDS, already on disk at `datasets/unsw_nb15/`, aligns with `network_v9_v3` checkpoint.
- Audit `unsw_nb15` CSV for tactic / alert_type / campaign_id columns; identify the equivalent of NSL-KDD's three label tracks.
- Add YAML block with dev seed 42 + eval seed 142.
- Run the full baseline roster.
- **Decision gate:** if V3 wins by >0.1 ARI on at least one track, freeze v1.0 immediately and move on. If margin is <0.1 or V3 loses, open a brief investigation doc *before* sweeping — the finding itself is publishable.
- Expected effort: 1–2 days. Most risk is loader/label-column drift, not modeling.

### IV.3 TON_IoT
First non-network dataset. Tests whether heterogeneous-graph value claim survives a domain shift.
- Audit IoT/IIoT label schema (per-sensor and global).
- Expect to need a TON_IoT-specific loader because schema differs more from NSL-KDD.
- Honest expectation per CLAUDE.md and design history: V3 may *not* dominate here. Plan to publish the result either way.
- Expected effort: 2–3 days (loader is the heavy part).

### IV.4 CICIDS2017
Modern network IDS benchmark. Confirms the network-IDS family beyond NSL-KDD/UNSW.
- 433K rows — first dataset where full-corpus eval is expensive; default to a fixed 10K stratified eval subset, with an optional `--full-eval` flag.
- Expected effort: 1–2 days.

### IV.5 SQTK_SIEM
Heterogeneous SIEM. The dataset where heterogeneous graph structure is most likely to add real value — but also the smallest (5.1K rows).
- Use `siem_supcon_v4/best.pt` as the dataset-specific checkpoint; document the multi-checkpoint policy in `docs/datasets/sqtk_siem/protocol.md`.
- Expected effort: 1–2 days.

### IV.6 DARPA OpTC
Host EDR, 4.6M rows, binary attribution. Largest scale; primary metric is `binary_ARI`.
- Decide upfront: stratified subset vs streaming evaluation. Document choice in `protocol.md` *before* running.
- Use `multidomain_v2_optc_finetuned/best_supervised.pt` only if V3-unsupervised-safe path through it exists; otherwise stay on `network_v9_v3` and accept lower ARI honestly.
- Expected effort: 3–5 days (data scale is the cost).

### IV.7 (Future — separate epoch) DARPA TC3 + NODLINK
Provenance datasets that enable direct comparison to MAGIC / FLASH / ORTHRUS / NODLINK published numbers. Out of scope for this plan; tracked here as the natural extension once IV.2–IV.6 are frozen.

### IV.8 (Cross-cutting — runs after IV.2 lands) Heavy GNN baselines
Add HGT / HAN / R-GCN via PyG drop-ins, then MAGIC / FLASH ports. Once added, retroactively re-run all frozen datasets and produce **v1.1** baselines that include the heavier roster. v1.0 stays as the historical record.

---

## Part V — Documentation Discipline (ongoing)

### V.1 Single source of truth per dataset
`docs/datasets/<name>/v<X.Y>_baseline.md` is the only place that quotes specific numbers. README and IMPROVEMENT_LOG link to it, never duplicate it.

### V.2 Decision log discipline
Every non-trivial choice during a dataset's lifecycle gets a one-line entry in `decision_log.md`:
- Date, choice, alternatives considered, rationale, link to artifact that motivated it.
- Mirrors the Phase 3 sweep learning — that lesson cost real time; documenting it once saves it from being relearned per dataset.

### V.3 Manifest as contract
Every published result must point at a manifest JSON. PRs that change `benchmark/results/frozen/*` without a manifest update fail CI. Add `tests/test_frozen_results_have_manifests.py`.

### V.4 No silent retunes
Changing `engine_kwargs` for a frozen dataset is forbidden in-place. It produces a new version (v1.1, v2.0). The old frozen directory is never edited.

### V.5 Repo navigation doc
`docs/repo_layout.md` — a single short page showing the post-cleanup structure, what each top-level directory is for, and where new code/data/docs belong. The first thing a new contributor (or you, six months from now) reads.

---

## Critical Files Summary

**Created (or formalized) by this plan:**
- `benchmark/results/frozen/nsl_kdd/v1.0/{results,summary,manifest,engine_config,checkpoint_sha256,dataset_sha256,environment}.{csv,json,yaml,txt}`
- `docs/datasets/nsl_kdd/{README,v1.0_baseline,audit,protocol,decision_log,learnings}.md`
- `docs/datasets/<each_future_dataset>/...` (same skeleton)
- `docs/cleanup/dependency_map_v1.md`
- `docs/lessons/phase3_sweep_methodology.md`
- `docs/repo_layout.md`
- `archive/v2_legacy/{B1..B9}/...`
- `benchmark/clustering_sweep_full_engine.py`
- `tests/test_nsl_kdd_v1_frozen.py`
- `tests/test_frozen_results_have_manifests.py`
- (renamed) `benchmark/clustering_sweep_standalone.py`

**Touched:**
- `IMPROVEMENT_LOG.md` — converted to index format with per-dataset pointers
- `README.md` — link to `docs/repo_layout.md` and `docs/datasets/*/v*_baseline.md`
- `benchmark/datasets_real.yaml` — frozen NSL-KDD block + UNSW-NB15 block

**Not touched in this plan:**
- `mitre_core/` (no model behavior changes)
- `hgnn/hgnn_correlation.py` (guardrails already in place)
- `hgnn_checkpoints/network_v9_v3/network_it_best.pt` (the validated checkpoint)

---

## Verification (end-to-end)

1. `pytest tests/test_nsl_kdd_v1_frozen.py` — frozen NSL-KDD v1.0 reproduces exactly.
2. `pytest tests/test_frozen_results_have_manifests.py` — every frozen dir has a manifest.
3. `pytest tests/test_split_disjoint.py` — every dataset's dev/eval pairs are disjoint.
4. `pytest -q` post-cleanup — full suite passes after every quarantine batch.
5. `python -m benchmark.run_benchmark --config benchmark/datasets_real.yaml --eval-only` — produces results matching the frozen baseline within tolerance.
6. `docs/repo_layout.md` accurately describes the on-disk structure.
7. Each frozen dataset has a `vX.Y_baseline.md` answering the reproduction question without any other file.

---

## Non-Goals (explicit)

- No model retraining, no new SSL objectives, no encoder changes in this plan.
- No new datasets added beyond the six already on disk (DARPA TC3 / NODLINK explicitly deferred).
- No deletion of `hgnn_checkpoints/` — those are physical artifacts the freeze depends on.
- No restructuring of `mitre_core/` — it is already the V3 package surface.
- No paper / figure generation — happens once at least three datasets are frozen.

---

## Execution Order Summary

1. **Part I — Freeze NSL-KDD v1.0** (1 day; no risk; pure documentation + artifact copy)
2. **Part II — Cleanup batches B1 → B9** (2–3 days; each batch is one commit + test gate)
3. **Part III — Lifecycle scaffolding** (0.5 day; templates + reusable tests)
4. **Part IV.2 — UNSW-NB15** (1–2 days; first application of the lifecycle on a new dataset)
5. **Part IV.3 → IV.6** — TON_IoT, CICIDS2017, SQTK_SIEM, OpTC, one at a time, each ending in a vX.Y freeze.
6. **Part IV.8** — heavy GNN baselines pass; retro v1.1 across frozen datasets.
7. **Part IV.7** — DARPA TC3 + NODLINK (separate epoch).

---

## Part VI — Master Plan Freeze Ritual

**Goal:** lock this plan into the repo as an immutable reference so subplans cannot drift without an explicit, visible change.

### VI.1 Canonical location in the repo
The Master Plan freezes at:
- **`E:\Private\MITRE-CORE 2\MITRE-CORE_V3\docs\plans\MASTER_PLAN_v1.0.md`**

This file is copied verbatim from the planning file at `C:\Users\rahul\.claude\plans\mitre-core-v3-mitre-core-piped-liskov.md` at freeze time, then treated as read-only by convention. The planning file in `~/.claude/plans/` stays as the working draft; the repo file is the contract.

### VI.2 Freeze artifacts (committed together)
- `docs/plans/MASTER_PLAN_v1.0.md` — frozen master plan content
- `docs/plans/MASTER_PLAN_v1.0.sha256` — hash of the frozen plan content
- `docs/plans/README.md` — index: "v1.0 is the active master plan. Subplans live under `docs/datasets/<name>/subplan.md` and must declare which master-plan section they implement."
- Git tag: `master-plan-v1.0`

### VI.3 Alignment recheck protocol
Before starting any subplan, and again before freezing any dataset:
1. Open `docs/plans/MASTER_PLAN_v1.0.md`.
2. Confirm the subplan's scope, lifecycle stage, and exit criteria match a specific section in the master plan (cite the section number in the subplan).
3. If the subplan needs to deviate, **do not modify the master plan in place**. Instead, write `docs/plans/MASTER_PLAN_v1.1.md` with a diff summary at the top, leaving v1.0 immutable, and update `docs/plans/README.md` to point at v1.1 as active.
4. Add `tests/test_master_plan_unchanged.py` that asserts the SHA256 of `MASTER_PLAN_v1.0.md` matches `MASTER_PLAN_v1.0.sha256`. Any edit to v1.0 fails CI.

### VI.4 Subplan declaration requirement
Every per-dataset subplan must open with a 3-line header:
```
Master plan: MASTER_PLAN_v1.0.md
Section implemented: <e.g. "Part IV.2 — UNSW-NB15">
Lifecycle stages covered: <e.g. "1–6">
```
This makes drift detectable by `grep`, and forces the implementer to look at the master plan before writing code.

---

## Part VII — Subplan IV.2: UNSW-NB15 v1.0

> **Master plan:** `docs/plans/MASTER_PLAN_v1.0.md`
> **Section implemented:** Part IV.2 — UNSW-NB15
> **Lifecycle stages covered:** 1–6 (full lifecycle)

### VII.1 Why UNSW-NB15 first
- Closest analog to NSL-KDD: network IDS, MITRE-format CSV already on disk at `datasets/unsw_nb15/mitre_format.csv`, aligns with the validated `network_v9_v3/network_it_best.pt` checkpoint.
- Confirms the NSL-KDD lifecycle is reusable, not bespoke. If something breaks here, the lifecycle template needs fixing — better to find that now than on dataset #5.

### VII.2 Subplan location
`docs/datasets/unsw_nb15/subplan.md` — created as the first artifact of this subplan. All subsequent UNSW work is logged in the same folder (`audit.md`, `protocol.md`, `decision_log.md`, `learnings.md`, `v1.0_baseline.md`).

### VII.3 Stage-by-stage execution

#### Stage 1 — Audit
- Create `scripts/audit_unsw_nb15_labels.py` (mirror of `scripts/audit_nsl_kdd_labels.py`).
- Inputs: `datasets/unsw_nb15/mitre_format.csv`.
- Outputs (written to `docs/datasets/unsw_nb15/audit.md`):
  - row count, SHA256 hash
  - presence and value counts of: `tactic`, `alert_type`, `campaign_id`, `stage`, any UNSW-native label column (`attack_cat`, `label`)
  - flag tactics that merge >3 dissimilar attack categories or have <20 rows in any candidate 10K subset
  - candidate label tracks for benchmark inclusion
- **Exit criterion:** at least one usable multi-class track exists. If the only viable label is binary `alert_type`, document that as a finding and continue — UNSW becomes a binary-headline dataset.

#### Stage 2 — Protocol Freeze
- Decide sample size: default 10,000 to match NSL-KDD. If UNSW row count >> NSL-KDD, optionally raise to 20,000 and document the choice in `decision_log.md`.
- Add YAML block to `benchmark/datasets_real.yaml`:
  - `UNSW-NB15-dev` — `sample_seed: 42`
  - `UNSW-NB15` — `sample_seed: 142`, `exclude_sample_seeds: [42]`
  - `alt_label_cols` populated from audit findings
  - Same `engine_kwargs` as NSL-KDD (single layer, pca=16, eps=0.1, mcs=5) — no per-dataset retune at this stage
- Run `python -m benchmark.run_benchmark --dataset UNSW-NB15-dev --eval-only` to persist `benchmark/splits/unsw_nb15_10000_seed{42,142}.{npy,json}` and verify wiring.
- **Exit criterion:** splits persisted, disjoint, hashes recorded; manifest emitted.

#### Stage 3 — Baseline Roster
- Run all 10 baselines + V3 on the eval subset, 3 seeds:
  - raw: K-Means, DBSCAN, HDBSCAN, Spectral (raw), PCA+K-Means, PCA+HDBSCAN
  - embedding-based: K-Means (emb), Spectral (emb), HDBSCAN (emb)
  - V3: MITRE-CORE V3 with current engine_kwargs
- Output: `benchmark/results/latest/unsw_nb15/baseline_roster.csv` + `_summary.csv` + `_manifest.json`.
- **Exit criterion:** all rows produced; no method crashed; metrics within expected ranges (not all 0, not all 1).

#### Stage 4 — Decision Gate
Compare V3 ARI on each label track against the best baseline:
- **Path A (V3 wins by >0.1 ARI on at least one track):** skip sweep, proceed to Stage 5.
- **Path B (margin <0.1 or V3 loses):** write `docs/datasets/unsw_nb15/investigation.md` before any retuning. Investigate first:
  - graph audit: edge counts per type, connected-component distribution, oversmoothing risk
  - feature audit: degenerate columns, scale mismatches
  - label audit: is the winning baseline picking up label structure V3 misses?
  Then, and only then, consider a full-engine sweep via `benchmark/clustering_sweep_full_engine.py` (do not use the standalone sweep — Phase 3 NSL-KDD lesson).
- Record the decision and rationale in `decision_log.md`.

#### Stage 5 — Metrics + Demotion
- Confirm the full 12-metric set is emitted, including `attack_f1_demoted`, `n_pred_clusters`, `noise_fraction`, `coverage`, `dominant_confusion_accuracy`.
- Flag any metric that is degenerate on UNSW (e.g., constant across methods) — demote in the dataset doc the same way `attack_f1` was demoted on NSL-KDD.

#### Stage 6 — Freeze v1.0
- Copy artifacts to `benchmark/results/frozen/unsw_nb15/v1.0/` (same file set as NSL-KDD freeze).
- Write `docs/datasets/unsw_nb15/v1.0_baseline.md` from the template established by `docs/datasets/nsl_kdd/v1.0_baseline.md`.
- Add `tests/test_unsw_nb15_v1_frozen.py` (parameterize the NSL-KDD test if possible to avoid copy-paste).
- Git tag `unsw-nb15-v1.0`.
- Update `IMPROVEMENT_LOG.md` index with a one-line entry pointing at the new baseline doc.
- Update `docs/datasets/unsw_nb15/learnings.md` with anything that should change the lifecycle template before TON_IoT.

### VII.4 Files created or touched

**New:**
- `docs/datasets/unsw_nb15/{README,subplan,audit,protocol,decision_log,learnings,v1.0_baseline}.md`
- `docs/datasets/unsw_nb15/investigation.md` (only if Path B in Stage 4)
- `scripts/audit_unsw_nb15_labels.py`
- `benchmark/splits/unsw_nb15_10000_seed{42,142}.{npy,json}`
- `benchmark/results/latest/unsw_nb15/*` (working)
- `benchmark/results/frozen/unsw_nb15/v1.0/*` (immutable at end)
- `tests/test_unsw_nb15_v1_frozen.py`

**Touched:**
- `benchmark/datasets_real.yaml` — UNSW-NB15-dev + UNSW-NB15 blocks
- `IMPROVEMENT_LOG.md` — index entry
- `docs/plans/README.md` — pointer to active subplan

**Not touched:**
- Master plan (`docs/plans/MASTER_PLAN_v1.0.md`) — frozen
- NSL-KDD frozen artifacts (`benchmark/results/frozen/nsl_kdd/v1.0/`) — frozen
- `mitre_core/`, `hgnn/`, model behavior — unchanged

### VII.5 Subplan exit criteria
This subplan is complete when:
1. `pytest tests/test_unsw_nb15_v1_frozen.py` passes.
2. `pytest tests/test_master_plan_unchanged.py` passes (no drift).
3. `pytest tests/test_nsl_kdd_v1_frozen.py` still passes (no regression).
4. `docs/datasets/unsw_nb15/v1.0_baseline.md` answers reproduction without any other file.
5. Git tag `unsw-nb15-v1.0` exists at the freeze commit.

### VII.6 Risks and mitigations
- **Label-column drift** between UNSW and NSL-KDD MITRE conversions → Stage 1 audit catches this before any benchmark runs.
- **V3 loses on UNSW** → Path B investigation, documented honestly. Not a failure of the subplan; the finding becomes the story.
- **Lifecycle template gaps** discovered during Stage 6 → write the fix to `docs/datasets/_template/` rather than retro-editing NSL-KDD's docs.
