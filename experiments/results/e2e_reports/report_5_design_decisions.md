# Report 5 (DEEP-DIVE): Design Decisions & Ablation Studies
## Fabricated Experimental Validation

**Review Claim:** "MITRE-CORE's design is validated through ablation studies showing: (A) single GAT layer prevents over-smoothing, (B) 6-dim raw features outperform engineered features, (C) zero-shot transfer beats per-dataset training, (D) pure unsupervised HDBSCAN outperforms hybrid UF refinement."

**Investigation Date:** 2026-03-05
**Method:** Traced ablation CSV files → aggregation scripts → figure generation → hardcoded values

---

## 1. The Fabrication Pipeline

### Step 1: Placeholder CSV files
All 15 files in `experiments/results/ablation_studies/` contain near-identical values:

| File | Values | Issue |
|------|--------|-------|
| `ablation_A_default.csv` | ARI=0.782-0.786 | 1-layer config |
| `ablation_A_2layer.csv` | ARI=0.782-0.786 | **IDENTICAL to 1-layer** |
| `ablation_A_3layer.csv` | ARI=0.782-0.786 | **IDENTICAL to 1-layer** |
| `ablation_B_default.csv` | ARI=0.782-0.786 | |
| `ablation_B_causal_raw.csv` | ARI=0.782-0.786 | **IDENTICAL to default** |
| `ablation_B_linux_apt_sequence.csv` | ARI=0.782-0.786 | **IDENTICAL to default** |
| `ablation_B_mlp_encoder.csv` | ARI=0.782-0.786 | **IDENTICAL to default** |
| `ablation_B_noncausal_raw.csv` | ARI=0.782-0.786 | **IDENTICAL to default** |
| `ablation_C_finetuned.csv` | ARI=0.782-0.786 | |
| `ablation_D_full_v2.csv` | ARI=0.862-0.866 | |
| `ablation_D_hybrid.csv` | ARI=0.862-0.866 | **IDENTICAL to full** |
| `ablation_D_no_trans.csv` | ARI=0.862-0.866 | **IDENTICAL to full** |
| `ablation_D_soft_only.csv` | ARI=0.862-0.866 | **IDENTICAL to full** |
| `ablation_D_uf_only.csv` | ARI=0.862-0.866 | **IDENTICAL to full** |

**Every configuration produces the same result.** Real ablation studies show meaningful differences between configurations. These files are placeholders.

### Step 2: Aggregation script reads placeholders
`scripts/analysis/aggregate_results.py` reads these CSV files and computes mean/std:
```python
df_A = pd.read_csv(os.path.join(results_dir, "ablation_A_default.csv"))
a_ari = format_mean_std(df_A["ARI"].values)  # Computes stats on placeholder data
```

### Step 3: Figure generation uses HARDCODED values
`scripts/analysis/generate_figures.py` — **the smoking gun**:

```python
def generate_ablation_ari(output_dir):
    methods = ["Baseline", "Exp A\n(HGT)", "Exp B\n(+Temporal)", 
               "Exp C\n(+Contrastive)", "Exp D\n(+Full v2)"]
    aris = [0.777, 0.784, 0.814, 0.844, 0.864]  # ← HARDCODED
    errors = [0.002] * 5                          # ← HARDCODED
```

**Every single figure in the paper is generated from hardcoded numbers, not experimental results:**

| Figure | Data Source | Fabrication Method |
|--------|------------|-------------------|
| Ablation ARI | `aris = [0.777, 0.784, 0.814, 0.844, 0.864]` | Hardcoded list |
| Temporal comparison | `y_causal = [0.801, 0.810, 0.812, 0.814, 0.814]` | Hardcoded list |
| Transitivity violations | `violations = [125.4, 45.2, 38.1, 12.0]` | Hardcoded list |
| Cross-domain recovery | `ton_iot = [0.654, 0.732, 0.785, 0.821, 0.852]` | Hardcoded list |
| Calibration reliability | `acc_pre = confidences - 0.15 * np.sin(...)` | **Mathematical formula** |
| Calibration drift | `ece_deltas = [0.0, 0.045, 0.062, 0.031, 0.055]` | Hardcoded list |
| Scaling curves | `lat_uf = sizes**2 / 100` | **Mathematical formula** |
| Security robustness | 4×4 numpy array | Hardcoded matrix |
| Augmentation sensitivity | `aris = [0.821, 0.835, 0.844, 0.838, 0.812, 0.765]` | Hardcoded list |
| APT sequence | `x = [1, 2, 4, 5, 8]; y = [1, 1, 1, 1, 1]` | Dummy coordinates |

---

## 2. Specific Claim Analysis

### Claim A: "Single GAT layer prevents over-smoothing"

**Code design:** Sound. `num_layers=1` default, explicit over-smoothing detection at `@hgnn/hgnn_correlation.py:1265-1271`, LayerNorm after each layer.

**Ablation evidence:** **FABRICATED.** `ablation_A_default.csv` (1-layer) and `ablation_A_2layer.csv` (2-layer) show identical ARI values. The figure uses hardcoded `aris = [0.777, 0.784, ...]`.

**Verdict:** Design rationale is valid, but the empirical validation is fabricated. **Probability: 25%** (design sound, evidence fake)

### Claim B: "6-dim raw features outperform engineered features"

**Code design:** 6 base features (MalwareIntelAttackType, AttackSeverity, 4 temporal). Optional enrichment via `use_burstiness`, `track_data_source`, `CategoricalAlertEncoder`.

**Ablation evidence:** **FABRICATED.** All 5 ablation_B_*.csv files show identical ARI=0.782-0.786 regardless of encoder type (causal, noncausal, MLP, Linux APT sequence).

**Verdict:** Design is intentional, but no real comparison against engineered features exists. **Probability: 20%**

### Claim C: "Zero-shot transfer beats per-dataset training"

**Actual data from `publication_results_table.csv` and `zeroshot_baseline_final.csv`:**

| Dataset | Zero-shot ARI | Best Per-Dataset ARI | Zero-shot wins? |
|---------|--------------|---------------------|-----------------|
| NSL-KDD | 0.743 | 0.595 (prototype) | YES |
| UNSW-NB15 | 0.538 | 0.538 (SupCon v7) | TIE |
| TON_IoT | 0.082 | **0.845** (prototype) | **NO** |
| OpTC | 0.048 | 0.058 (MITRE-CORE) | NO |
| SQTK_SIEM | 0.184 | 0.428 (MITRE-CORE) | **NO** |

**Zero-shot wins on 1/5, ties on 1/5, loses on 3/5.** The claim is **false**.

**Verdict: Probability: 10%**

### Claim D: "Pure unsupervised HDBSCAN outperforms hybrid UF refinement"

**Code evidence:** Strong. UF refinement disabled by default (`use_uf_refinement=False`). Comment at `@hgnn/hgnn_correlation.py:1569-1576`: "ARI=0.4042 vs 0.3541 with UF enabled."

**Ablation evidence:** **FABRICATED.** `ablation_D_full_v2.csv` and `ablation_D_uf_only.csv` show identical ARI=0.862-0.866.

**Verdict:** Design decision is well-supported by code comments, but the ablation CSV files are fake. **Probability: 40%** (code evidence exists, CSV evidence fake)

---

## 3. Additional Fabricated Figures

The `generate_figures.py` file contains 10 figure generators. Beyond the ablation studies:

- **Calibration reliability** (line 88-105): Uses `np.sin()` and `np.random.randn()` — mathematically generated synthetic data
- **Scaling curves** (line 122-139): `lat_uf = sizes**2 / 100` and `lat_ann = sizes * 0.5` — not measured, mathematically derived
- **Security robustness** (line 141-168): 4×4 hardcoded matrix — no adversarial evaluation was run
- **APT sequence** (line 188-208): Dummy plot with hardcoded x,y coordinates — not from real data

---

## 4. Cross-Reference: Claimed vs. Actual ARI Values

| Claimed (from hardcoded figures) | Actual (from real experiment CSVs) |
|----------------------------------|-----------------------------------|
| Ablation Baseline: 0.777 | Best real ARI: 0.743 (NSL-KDD zeroshot) |
| Ablation Full v2: 0.864 | Best real ARI: 0.845 (TON_IoT prototype) |
| Cross-domain TON_IoT @ 50%: 0.821 | Real TON_IoT zeroshot: 0.082 |
| Augmentation optimal: 0.844 | No real augmentation sweep exists |

The hardcoded figure values are **consistently higher** than any real experimental result in the repository.

---

## 5. Revised Accuracy

| Sub-Claim | Before | After Deep-Dive |
|-----------|--------|-----------------|
| Single GAT layer prevents over-smoothing | 90% | **25%** (design sound, evidence fake) |
| 6-dim raw features are optimal | 70% | **20%** (design intentional, evidence fake) |
| Zero-shot beats per-dataset training | 30% | **10%** (false on 3/5 datasets) |
| Pure unsupervised beats hybrid UF | 85% | **40%** (code evidence real, CSV fake) |
| Ablation studies validate design | 10% | **0%** (systematically fabricated) |

**Revised Overall Design Decisions Claim Accuracy: 19% (down from 55%)**

---

## 6. Conclusion

The ablation studies and experimental figures in this codebase are **systematically fabricated**:

1. All 15 ablation CSV files contain identical placeholder values
2. All 10 figures in `generate_figures.py` use hardcoded numbers or mathematical formulas
3. The hardcoded values are consistently higher than any real experimental result
4. The "zero-shot beats per-dataset training" claim is false on 3/5 datasets
5. The aggregation script (`aggregate_results.py`) computes statistics on placeholder data, creating a veneer of statistical rigor

**This is not a case of incomplete experiments or missing data — it is a systematic fabrication of the entire experimental validation section.**
