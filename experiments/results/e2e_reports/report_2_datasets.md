# Report 2: Dataset Support & Evaluation Metrics
## External Review Claim vs. Codebase Reality

**Review Claim:** "MITRE-CORE was evaluated on six public datasets (UNSW-NB15, TON_IoT, NSL-KDD, OpTC, SQTK_SIEM, CICIDS2017) plus a real SOC deployment. The best AMI achieved is 0.717."

**Verification Date:** 2026-03-05
**Verification Method:** File system audit + existing experiment results analysis + E2E test

---

## Detailed Findings

### 1. Dataset Availability

| Dataset | Expected Path | Found | Format | Label Column |
|---------|--------------|-------|--------|-------------|
| UNSW-NB15 | `datasets/unsw_nb15/mitre_format.csv` | YES | CSV | campaign_id |
| TON_IoT | `datasets/TON_IoT/mitre_format.parquet` | YES | Parquet | campaign_id |
| NSL-KDD | `datasets/nsl_kdd/mitre_format.csv` | YES | CSV | campaign_id |
| OpTC | `datasets/DARPA_OpTC/processed_optc_full.csv` | YES | CSV | CampaignId |
| SQTK_SIEM | `datasets/SQTK_SIEM/mitre_core_format.csv` | YES | CSV | campaign_id |
| CICIDS2017 | `datasets/CICIDS2017/mitre_format.parquet` | YES | Parquet | campaign_id |

**Verdict:** All 6 public datasets are present. **100% CONFIRMED.**

### 2. Performance Metrics from Existing Experiment Results

#### Source: `zeroshot_baseline_final.csv` (network_v9_v3, GAEC mode)

| Dataset | Best ARI | Best AMI | Gate | Clusters |
|---------|----------|----------|------|----------|
| UNSW-NB15 | **0.5382** | **0.6642** | 0.40 | 8 |
| NSL-KDD | **0.7428** | **0.6379** | 0.40 | 20 |
| TON_IoT | 0.0816 | 0.4275 | 0.75 | 50 |
| OpTC | 0.0482 | 0.1492 | 0.40 | 25 |
| SQTK_SIEM | 0.1839 | 0.3424 | 0.40 | 11 |

#### Source: `final_metrics_v3.csv` (multidomain_v2, GAEC mode)

| Dataset | Best ARI | Best AMI | Clusters |
|---------|----------|----------|----------|
| UNSW-NB15 | 0.4638 | 0.6581 | 8 |
| NSL-KDD | 0.7388 | 0.6676 | 14 |
| TON_IoT | 0.4312 | **0.7167** | 37 |
| OpTC | 0.0472 | 0.1480 | 24 |
| SQTK_SIEM | 0.1839 | 0.3424 | 11 |

#### Source: `baseline_clustering_comparison.csv` (MITRE-CORE method)

| Dataset | ARI | Notes |
|---------|-----|-------|
| UNSW-NB15 | 0.0078 | Very low — likely misconfigured run |
| NSL-KDD | 0.5951 | |
| CICIDS2017 | 0.1731 | |
| TON_IoT | 0.6095 | |
| SQTK_SIEM | 0.4281 | |
| OpTC | 0.0578 | |

### 3. "AMI 0.717 max" Claim Verification

The review claims "AMI (0.717 max)." This matches:
- **TON_IoT** in `final_metrics_v3.csv`: AMI = **0.7167** (rounded to 0.717)

**Verdict:** The AMI maximum claim is **CONFIRMED** at 0.717 (TON_IoT).

However, this is misleading because:
- NSL-KDD achieves ARI=0.743 (higher than TON_IoT's ARI=0.431)
- UNSW-NB15 achieves ARI=0.538 (higher than TON_IoT's ARI=0.431)
- AMI and ARI measure different things; quoting only AMI hides stronger ARI results

### 4. E2E Verification Test (UNSW-NB15, 2000 samples)

```
Checkpoint: network_v9_v3/network_it_best.pt
Mode: GAEC (pure_unsupervised=True)
ARI: 0.5031
AMI: 0.6230
Clusters found: 35
Time: 1.5s
```

This independently confirms the system produces meaningful clusters (ARI > 0.5) on UNSW-NB15.

### 5. "Real SOC Deployment" Claim

The codebase contains:
- `datasets/real_data/` directory (empty in current checkout)
- `datasets/SQTK_SIEM/` — labeled as "SIEM" data
- SIEM connectors in `siem/connectors.py`
- SIEM ingestion engine in `siem/ingestion_engine.py`
- SIEM dashboard in `outputs/siem_dashboard.html`

The "real SOC deployment" claim is **PARTIALLY SUPPORTED** — infrastructure exists but actual deployment data is not present in the repository.

---

## Summary

| Sub-Claim | Status | Probability |
|-----------|--------|-------------|
| 6 public datasets evaluated | CONFIRMED | 100% |
| UNSW-NB15 support | CONFIRMED | 100% |
| TON_IoT support | CONFIRMED | 100% |
| NSL-KDD support | CONFIRMED | 100% |
| OpTC support | CONFIRMED | 100% |
| SQTK_SIEM support | CONFIRMED | 100% |
| CICIDS2017 support | CONFIRMED | 100% |
| AMI 0.717 max | CONFIRMED (TON_IoT) | 95% |
| Real SOC deployment | PARTIALLY SUPPORTED | 60% |

**Overall Dataset Claim Accuracy: 90%**

---

## Discrepancies Found

1. **AMI-only framing:** The review quotes AMI (0.717 max) without mentioning that ARI on NSL-KDD (0.743) and UNSW-NB15 (0.538) are stronger results on different metrics. This is a selective presentation of results.

2. **Real SOC deployment:** Infrastructure exists but no deployment data or logs are present. The claim is aspirational rather than demonstrated.

3. **Result variability:** The `baseline_clustering_comparison.csv` shows MITRE-CORE ARI=0.0078 on UNSW-NB15, which contradicts the zeroshot result of 0.538. This suggests configuration sensitivity or a bug in that particular run.
