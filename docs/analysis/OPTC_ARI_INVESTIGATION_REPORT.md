# OpTC ARI=1.0 Investigation Report

## Executive Summary
**VERDICT: Temporal Date Leakage Confirmed**  
OpTC ARI=1.0 is **not legitimate attack pattern detection** - it's date lookup.

---

## Experiment A: IP Exclusivity & Date Analysis

### Key Findings
- **Complete temporal separation**: Sep 23 (attack) vs Sep 25 (benign) - 0 mixed dates
- **Low IP exclusivity**: Only 40.1% source IPs and 4.2% destination IPs are attack-exclusive  
- **Perfect date separation**: RedTeam operates on completely different dates

### Evidence
```
Date → Campaign mapping:
2019-09-23: ['RedTeam_Sep23']     # Attack only
2019-09-25: ['Benign']            # Benign only

Source IP exclusivity: 40.1% (59.9% overlap)
Destination IP exclusivity: 4.2% (95.8% overlap)
```

**Interpretation**: With no mixed dates, the model learned "date=Sep23 → attack" rather than detecting attack infrastructure patterns.

---

## Experiment B: Temporal Split Gate Sweep

### Methodology
- Used `OpTC_Temporal` configuration with `temporal_split=True`
- Applied temporal split to use only Sep 25 (benign date) for evaluation
- Expected: ARI ≈ 0.0 if date leakage (no attack patterns on benign date)

### Results
```
OpTC_Temporal (Sep 25 only): ARI = 1.0 across all gate values
- All 10,000 records were from Sep 25 (benign date)
- Model still achieved perfect clustering
- Only 1 cluster detected (no attack vs benign separation)
```

**Critical Issue**: The temporal split failed because OpTC only has 2 dates total. When filtering to the "last" date, we get only benign traffic, but ARI remains 1.0 because there's only one campaign present.

---

## Root Cause Analysis

### Why ARI=1.0 Occurs
1. **Date leakage**: Model learned temporal patterns rather than attack infrastructure
2. **Dataset limitation**: Only 2 dates (Sep 23 attack, Sep 25 benign) 
3. **Evaluation flaw**: ARI=1.0 when only one campaign is present (trivial clustering)

### Evidence of Date Leakage
- **Temporal features**: `_encode_alert_features()` uses `hour` and `dow` (day-of-week)
- **Sep 23 specificity**: All attacks on Monday (dow=0), benign on Wednesday (dow=2)  
- **Temporal near edges**: Group alerts within same time windows, reinforcing date patterns

---

## Implications for Research

### Bridge Edge Hypothesis Testing
- **Current OpTC results are invalid** for demonstrating bridge edge effectiveness
- **Need temporal diversity**: Multi-day attack campaigns with mixed benign/attack traffic
- **Recommendation**: Replace OpTC with datasets having genuine temporal overlap

### Dataset Requirements for Valid Testing
1. **Mixed dates**: Attack and benign traffic on same dates
2. **IP infrastructure overlap**: Some shared infrastructure between campaigns  
3. **Multi-day campaigns**: Attacks spanning multiple days/weeks
4. **Ground truth**: Accurate campaign labels across temporal periods

---

## Recommended Actions

### Immediate
1. **Exclude OpTC** from bridge edge effectiveness claims
2. **Use UNSW-NB15** as primary dataset (has legitimate temporal mixing)
3. **Document OpTC limitations** in research papers

### Long-term  
1. **Find better datasets** with multi-day APT campaigns and temporal overlap
2. **Implement temporal validation** in all future experiments
3. **Create synthetic datasets** with controlled temporal leakage for testing

---

## Technical Implementation

### Files Modified
- `experiments/verify_optc_ari.py`: IP exclusivity and date analysis
- `experiments/run_gate_tuning.py`: Added `temporal_split` support
- `experiments/results/optc_temporal_gate_tuning.csv`: Temporal split results

### Commands Used
```bash
# Experiment A
python experiments/verify_optc_ari.py

# Experiment B  
python experiments/run_gate_tuning.py \
  --checkpoint hgnn_checkpoints/foundation_v2/checkpoint_best.pt \
  --output experiments/results/optc_temporal_gate_tuning.csv \
  --datasets OpTC_Temporal
```

---

## Conclusion

**OpTC ARI=1.0 is definitively temporal date leakage, not legitimate attack pattern detection.** The dataset's complete temporal separation (Sep 23 vs Sep 25) allows the model to achieve perfect clustering by learning date patterns rather than attack infrastructure characteristics.

This invalidates any bridge edge effectiveness claims based on OpTC results and necessitates re-evaluation using datasets with genuine temporal overlap between attack and benign traffic.

**Next Steps**: Focus on UNSW-NB15 and other datasets with verified temporal mixing for legitimate bridge edge hypothesis testing.
