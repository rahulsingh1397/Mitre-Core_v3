#!/usr/bin/env python3
# HISTORICAL: Entity collapse ablation. Confirmed zero effect (v2.25, Apr 19 2026).
# Do not re-run. Results archived. See MEMORY.md v2.25.
# HISTORICAL: entity collapse ablation. Negative result. See MEMORY.md v2.25.
"""
Entity Collapse Ablation Study Analysis

Analyzes the 7-seed entity collapse ablation study to determine whether
entity collapse (IP->host routing) improves clustering performance.

Usage:
    python experiments/analyze_entity_collapse.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import json
from pathlib import Path
import glob

def analyze_entity_collapse_ablation():
    """Analyze entity collapse ablation study results."""
    
    print("Analyzing entity collapse ablation study...")
    
    # Load results from the 7-seed study
    with_pattern = "experiments/results/collapse_with_seed*.csv"
    without_pattern = "experiments/results/collapse_without_seed*.csv"
    
    with_files = glob.glob(with_pattern)
    without_files = glob.glob(without_pattern)
    
    print(f"Found {len(with_files)} 'with entity collapse' files")
    print(f"Found {len(without_files)} 'without entity collapse' files")
    
    if len(with_files) == 0 or len(without_files) == 0:
        print("Error: Missing result files")
        return None
    
    # Load and combine results
    with_results = []
    without_results = []
    
    for file in sorted(with_files):
        df = pd.read_csv(file)
        # Get the best gate result (gate_value=0.65)
        best_row = df[df["gate_value"] == 0.65].iloc[0]
        seed = int(file.split("seed")[-1].split(".")[0])
        best_row["seed"] = seed
        with_results.append(best_row)
    
    for file in sorted(without_files):
        df = pd.read_csv(file)
        # Get the best gate result (gate_value=0.65)
        best_row = df[df["gate_value"] == 0.65].iloc[0]
        seed = int(file.split("seed")[-1].split(".")[0])
        best_row["seed"] = seed
        without_results.append(best_row)
    
    with_df = pd.DataFrame(with_results)
    without_df = pd.DataFrame(without_results)
    
    # Ensure we have paired results by seed
    merged_df = pd.merge(with_df, without_df, on="seed", suffixes=("_with", "_without"))
    
    print(f"Paired results: {len(merged_df)} seeds")
    
    if len(merged_df) == 0:
        print("Error: No paired results found")
        return None
    
    # Extract ARI and NMI values
    ari_with = merged_df["ari_with"].values
    ari_without = merged_df["ari_without"].values
    nmi_with = merged_df["nmi_with"].values
    nmi_without = merged_df["nmi_without"].values
    
    # Basic statistics
    analysis = {
        "study_info": {
            "n_seeds": len(merged_df),
            "checkpoint": "network_v9_v3",
            "gate_value": 0.65,
            "dataset": "OpTC",
            "hypothesis": "Entity collapse improves clustering by creating shared host neighbors"
        },
        "performance_comparison": {
            "ari_with_entity_collapse": {
                "mean": float(np.mean(ari_with)),
                "std": float(np.std(ari_with)),
                "values": [float(x) for x in ari_with]
            },
            "ari_without_entity_collapse": {
                "mean": float(np.mean(ari_without)),
                "std": float(np.std(ari_without)),
                "values": [float(x) for x in ari_without]
            },
            "nmi_with_entity_collapse": {
                "mean": float(np.mean(nmi_with)),
                "std": float(np.std(nmi_with)),
                "values": [float(x) for x in nmi_with]
            },
            "nmi_without_entity_collapse": {
                "mean": float(np.mean(nmi_without)),
                "std": float(np.std(nmi_without)),
                "values": [float(x) for x in nmi_without]
            }
        }
    }
    
    # Paired statistical tests
    # ARI paired t-test
    t_stat_ari, p_val_ari = stats.ttest_rel(ari_with, ari_without)
    
    # NMI paired t-test  
    t_stat_nmi, p_val_nmi = stats.ttest_rel(nmi_with, nmi_without)
    
    # Wilcoxon signed-rank test (non-parametric paired alternative)
    wilcoxon_ari, p_val_wilcoxon_ari = stats.wilcoxon(ari_with, ari_without)
    
    analysis["statistical_tests"] = {
        "ari_paired_ttest": {
            "t_statistic": float(t_stat_ari),
            "p_value": float(p_val_ari),
            "significant": bool(p_val_ari < 0.05),
            "effect_size_cohen_d": float((np.mean(ari_with) - np.mean(ari_without)) / np.std(np.array(ari_with) - np.array(ari_without), ddof=1)) if np.std(np.array(ari_with) - np.array(ari_without), ddof=1) > 0 else 0.0
        },
        "nmi_paired_ttest": {
            "t_statistic": float(t_stat_nmi),
            "p_value": float(p_val_nmi),
            "significant": bool(p_val_nmi < 0.05)
        },
        "ari_wilcoxon": {
            "statistic": float(wilcoxon_ari),
            "p_value": float(p_val_wilcoxon_ari),
            "significant": bool(p_val_wilcoxon_ari < 0.05)
        }
    }
    
    # Effect interpretation
    ari_improvement = np.mean(ari_with) - np.mean(ari_without)
    nmi_improvement = np.mean(nmi_with) - np.mean(nmi_without)
    
    analysis["interpretation"] = {
        "ari_improvement": float(ari_improvement),
        "nmi_improvement": float(nmi_improvement),
        "conclusion": (
            "Entity collapse significantly improves clustering" if p_val_ari < 0.05 
            else "Entity collapse does not show statistically significant improvement"
        ),
        "effect_size_interpretation": (
            "Large effect" if abs(analysis["statistical_tests"]["ari_paired_ttest"]["effect_size_cohen_d"]) > 0.8
            else "Medium effect" if abs(analysis["statistical_tests"]["ari_paired_ttest"]["effect_size_cohen_d"]) > 0.5
            else "Small effect" if abs(analysis["statistical_tests"]["ari_paired_ttest"]["effect_size_cohen_d"]) > 0.2
            else "Negligible effect"
        ),
        "practical_significance": (
            "Meaningful improvement" if abs(ari_improvement) > 0.01
            else "Minimal impact"
        )
    }
    
    # Add individual seed comparisons
    analysis["seed_by_seed"] = []
    for _, row in merged_df.iterrows():
        analysis["seed_by_seed"].append({
            "seed": int(row["seed"]),
            "ari_with": float(row["ari_with"]),
            "ari_without": float(row["ari_without"]),
            "ari_diff": float(row["ari_with"] - row["ari_without"]),
            "nmi_with": float(row["nmi_with"]),
            "nmi_without": float(row["nmi_without"]),
            "nmi_diff": float(row["nmi_with"] - row["nmi_without"])
        })
    
    return analysis

def main():
    analysis = analyze_entity_collapse_ablation()
    
    if analysis is None:
        print("Analysis failed")
        return
    
    # Save results
    output_path = "experiments/results/entity_collapse_ablation_stats.json"
    with open(output_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("ENTITY COLLAPSE ABLATION STUDY RESULTS")
    print("="*60)
    
    perf = analysis["performance_comparison"]
    print(f"\nARI Performance:")
    print(f"  With entity collapse:    {perf['ari_with_entity_collapse']['mean']:.4f} ± {perf['ari_with_entity_collapse']['std']:.4f}")
    print(f"  Without entity collapse: {perf['ari_without_entity_collapse']['mean']:.4f} ± {perf['ari_without_entity_collapse']['std']:.4f}")
    print(f"  Improvement:             {analysis['interpretation']['ari_improvement']:.4f}")
    
    print(f"\nNMI Performance:")
    print(f"  With entity collapse:    {perf['nmi_with_entity_collapse']['mean']:.4f} ± {perf['nmi_with_entity_collapse']['std']:.4f}")
    print(f"  Without entity collapse: {perf['nmi_without_entity_collapse']['mean']:.4f} ± {perf['nmi_without_entity_collapse']['std']:.4f}")
    print(f"  Improvement:             {analysis['interpretation']['nmi_improvement']:.4f}")
    
    stats_results = analysis["statistical_tests"]
    print(f"\nStatistical Significance (paired by seed):")
    print(f"  ARI paired t-test: t={stats_results['ari_paired_ttest']['t_statistic']:.3f}, p={stats_results['ari_paired_ttest']['p_value']:.4f}")
    print(f"  Significance: {'YES' if stats_results['ari_paired_ttest']['significant'] else 'NO'} (p < 0.05)")
    print(f"  Effect size (Cohen's d): {stats_results['ari_paired_ttest']['effect_size_cohen_d']:.3f} ({analysis['interpretation']['effect_size_interpretation']})")
    
    print(f"\nConclusion: {analysis['interpretation']['conclusion']}")
    print(f"Practical significance: {analysis['interpretation']['practical_significance']}")
    
    print(f"\nSeed-by-seed comparison:")
    for seed_data in analysis["seed_by_seed"]:
        print(f"  Seed {seed_data['seed']}: ARI diff = {seed_data['ari_diff']:+.4f}")
    
    print("="*60)

if __name__ == "__main__":
    main()
