# DARPA OpTC Dataset

DARPA Operationally Transparent Cyber (OpTC) dataset integrated into the MITRE-CORE V3 benchmark.

## Overview

| Property | Value |
|----------|-------|
| Source | DARPA OpTC eCAR-Bro endpoint events + Bro network logs |
| Rows | 4,656,650 |
| Columns | 29 |
| Primary label | `CampaignId` (binary: Benign / RedTeam_Sep23) |
| Secondary label | `Tactic` (largely degenerate — ~1.6M Execution, rest NaN) |
| File | `datasets/DARPA_OpTC/processed_optc_full.csv` |
| Format | MITRE-CORE standard (loader already renames columns) |

## Label Distribution

- **Benign**: ~4.46M (95.8%)
- **RedTeam_Sep23**: ~195K (4.2%)

## Graph Richness

OpTC has the richest entity graph in the benchmark:

- `src_ip` (`SourceAddress`) ✓
- `dst_ip` (`DestinationAddress`) ✓
- `hostname` (`SourceHostName`) ✓
- `username` (`SourceUserName`) ✓

## Benchmark Configuration

See `benchmark/datasets_real.yaml` for the full OpTC benchmark block.

Key decisions:
- `n_clusters: 2` — binary dataset (exactly 2 campaigns)
- `sample_size: 10000` — standard; 4.65M >> 10K so disjoint dev/eval split works normally
- `stratified_sample: true` — ensures ~420 RedTeam rows per 10K split
- `clustering_method: hdbscan` — with GAEC confidence scoring

## Headline Metric

**binary_ARI** (not standard ARI) — standard ARI is structurally low on 2-class datasets because fine-grained sub-clustering within the two true campaigns produces chance-level ARI by design.

## Documents

- `audit.md` — dataset audit report
- `protocol.md` — benchmark protocol decisions
- `decision_log.md` — stage-by-stage decision record
- `v1.0_baseline.md` — frozen baseline results
- `subplan.md` — implementation subplan
- `learnings.md` — post-freeze learnings (filled at Stage 6)
