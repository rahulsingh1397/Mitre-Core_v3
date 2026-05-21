# DARPA OpTC Dataset Audit Report

## File Metadata

| Property | Value |
|----------|-------|
| File | datasets/DARPA_OpTC/processed_optc_full.csv |
| Row count | 4,656,650 (expected 4,656,650) |
| SHA256 | 461f9d872a6cc389cdc4b87d26eef47596aa1d97ff657dd9c04db22953a31fe0 |

## Column Names

Total columns: 29

`
SourceAddress, DestinationAddress, DeviceAddress,
SourceHostName, DestinationHostName, DeviceHostName,
SourceUserName, EndDate, MalwareIntelAttackType,
AttackSeverity, Is_Attack, AlertId, ProcessName,
ProcessId, CommandLine, FilePath, NetworkProtocol,
SourcePort, DestinationPort, Tactic, BroFlowId,
CampaignId, CampaignStage, CampaignDurationHours,
CampaignSize, CampaignSequence, timestamp, date,
data_source
`

## Schema Diff vs MITRE Standard

**Present MITRE-standard columns (16):**
- AlertId, AttackSeverity, CommandLine, DestinationAddress, DestinationHostName, DestinationPort, DeviceAddress, DeviceHostName, EndDate, FilePath, NetworkProtocol, ProcessId, ProcessName, SourceAddress, SourceHostName, SourcePort, SourceUserName, Tactic

**Extra columns (11):**
- BroFlowId, CampaignDurationHours, CampaignId, CampaignSequence, CampaignSize, CampaignStage, Is_Attack, MalwareIntelAttackType, data_source, date, 	imestamp

**Missing MITRE-standard columns (3):**
- lert_type, campaign_id (lowercase), stage

Note: CampaignId (capital C, capital I) is present and serves as the campaign label. The MITRE standard expects lowercase campaign_id, but the benchmark reads whatever column name is specified in label_col.

## campaign_id Value Counts

| Value | Count | Percentage |
|-------|-------|------------|
| Benign | 4,461,191 | 95.8% |
| RedTeam_Sep23 | 195,459 | 4.2% |

## tactic Value Distribution

| Value | Count | Percentage |
|-------|-------|------------|
| Execution | 1,652,604 | 35.5% |
| (NaN/empty) | ~3,004,046 | ~64.5% |

Note: Only 1 unique tactic value (Execution) is present in the non-null rows. The loader maps most eCAR event types to Execution by default. This makes the 	actic track degenerate for benchmarking purposes.

## Graph Feasibility

| Entity | Column | Unique Values | Non-null | Status |
|--------|--------|--------------|----------|--------|
| src_ip | SourceAddress | 1,001 | 4,656,650 | OK |
| dst_ip | DestinationAddress | 1,001 | 4,656,650 | OK |
| hostname | SourceHostName | 95 | 4,656,650 | OK |
| username | SourceUserName | 7 | 4,656,650 | OK |

**-> Richest graph in benchmark** (src_ip, dst_ip, hostname, username all present and non-null).

## Temporal Leakage Note

Attacks are confined to 2019-09-23/25. Stratified sampling by campaign_id mitigates temporal leakage by ensuring proportional representation of Benign and RedTeam rows in each split. ~420 RedTeam rows are expected in each 10K split (4.2% of 10,000).

## Checkpoint Decision

Read A confirmed that use_geometric_confidence is a runtime flag in engine_kwargs. When True, the engine uses GAEC/HDBSCAN on backbone embeddings before the classification head. When False, it uses softmax on cluster_logits. The checkpoint type does **not** bypass this flag — a supervised checkpoint loaded with use_geometric_confidence=True will have its backbone used for GAEC, and the cluster_classifier head is ignored entirely.

**Decision**: GAEC overrides -> checkpoint = hgnn_checkpoints/multidomain_v2_optc_finetuned/best_supervised.pt

Rationale: The multidomain_v2_optc_finetuned checkpoint was explicitly fine-tuned on OpTC data (val_ari peaked at 0.818). Since GAEC uses the backbone embeddings and ignores the classification head, this checkpoint's learned graph representations should be optimal for OpTC.

## binary_ARI Rationale

Standard ARI is structurally low on 2-class datasets because fine-grained sub-clustering within the two true campaigns produces chance-level ARI by design. inary_ari maps each predicted cluster to its majority ground-truth label (Benign vs RedTeam) before computing ARI, making it the headline metric for OpTC.
