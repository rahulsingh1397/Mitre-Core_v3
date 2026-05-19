# DEPRECATED: Used ttest_ind (wrong test) on n=9 gate values (not independent seeds).
# p=0.021 claim permanently retracted. See MEMORY.md v2.24.
# DEPRECATED: used ttest_ind on n=9 gate values. Stats retracted. See MEMORY.md v2.24.
"""
experiments/analyze_optc_bridge_edges.py
---------------------------------------
Analyze bridge edge impact on OpTC dataset clustering performance.
Compares clustering results with vs without bridge edge correlations.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_optc_bridge_edges(results_path: str = "experiments/results/gate_tuning_results_v17.csv"):
    """
    Analyze bridge edge impact on OpTC clustering performance.
    
    Args:
        results_path: Path to gate tuning results CSV
        
    Returns:
        Dictionary with analysis results and statistical tests
    """
    
    # Load results
    results = pd.read_csv(results_path)
    
    # Filter for OpTC dataset
    optc_results = results[results["dataset"] == "OpTC"].copy()
    
    if len(optc_results) == 0:
        print("❌ No OpTC results found in gate tuning data")
        return None
    
    print(f"📊 Analyzing {len(optc_results)} OpTC gate tuning results...")
    
    # Separate results with and without bridge edges
    with_edges = optc_results[optc_results["bridge_edge_pct"] > 0].copy()
    without_edges = optc_results[optc_results["bridge_edge_pct"] == 0].copy()
    
    print(f"   With bridge edges: {len(with_edges)} results")
    print(f"   Without bridge edges: {len(without_edges)} results")
    
    # Basic statistics
    analysis = {
        "dataset_info": {
            "total_results": len(optc_results),
            "with_edges": len(with_edges),
            "without_edges": len(without_edges),
            "bridge_edge_coverage": float(optc_results["bridge_edge_pct"].mean()),
            "unique_bridge_flows": int(optc_results["unique_bridge_flows"].max())
        }
    }
    
    # Performance comparison
    if len(with_edges) > 0 and len(without_edges) > 0:
        
        # ARI comparison
        ari_with = with_edges["ari"].values
        ari_without = without_edges["ari"].values
        
        analysis["ari_comparison"] = {
            "with_edges_mean": float(np.mean(ari_with)),
            "without_edges_mean": float(np.mean(ari_without)),
            "improvement": float(np.mean(ari_with) - np.mean(ari_without)),
            "with_edges_std": float(np.std(ari_with)),
            "without_edges_std": float(np.std(ari_without))
        }
        
        # NMI comparison
        nmi_with = with_edges["nmi"].values
        nmi_without = without_edges["nmi"].values
        
        analysis["nmi_comparison"] = {
            "with_edges_mean": float(np.mean(nmi_with)),
            "without_edges_mean": float(np.mean(nmi_without)),
            "improvement": float(np.mean(nmi_with) - np.mean(nmi_without))
        }
        
        # Statistical significance tests (paired by seed)
        # Paired t-test for ARI
        t_stat_ari, p_val_ari = stats.ttest_rel(ari_with, ari_without)
        analysis["statistical_tests"] = {
            "ari_ttest": {
                "t_statistic": float(t_stat_ari),
                "p_value": float(p_val_ari),
                "significant": p_val_ari < 0.05,
                "effect_size": float((np.mean(ari_with) - np.mean(ari_without)) / np.std(np.array(ari_with) - np.array(ari_without), ddof=1))  # Cohen's d for paired samples
            }
        }
        
        # Paired t-test for NMI
        t_stat_nmi, p_val_nmi = stats.ttest_rel(nmi_with, nmi_without)
        analysis["statistical_tests"]["nmi_ttest"] = {
            "t_statistic": float(t_stat_nmi),
            "p_value": float(p_val_nmi),
            "significant": p_val_nmi < 0.05
        }
        
        # Mann-Whitney U test (non-parametric alternative)
        u_stat_ari, p_val_ari_mw = stats.mannwhitneyu(ari_with, ari_without, alternative='greater')
        analysis["statistical_tests"]["ari_mann_whitney"] = {
            "u_statistic": float(u_stat_ari),
            "p_value": float(p_val_ari_mw),
            "significant": p_val_ari_mw < 0.05
        }
        
        print(f"📈 ARI: {np.mean(ari_with):.4f} (with) vs {np.mean(ari_without):.4f} (without)")
        print(f"   Improvement: {np.mean(ari_with) - np.mean(ari_without):.4f}")
        print(f"   P-value: {p_val_ari:.4f} ({'significant' if p_val_ari < 0.05 else 'not significant'})")
        
        print(f"📈 NMI: {np.mean(nmi_with):.4f} (with) vs {np.mean(nmi_without):.4f} (without)")
        print(f"   Improvement: {np.mean(nmi_with) - np.mean(nmi_without):.4f}")
        print(f"   P-value: {p_val_nmi:.4f} ({'significant' if p_val_nmi < 0.05 else 'not significant'})")
    
    # Gate value analysis
    analysis["gate_analysis"] = {
        "best_gate_with_edges": float(with_edges.loc[with_edges["ari"].idxmax(), "gate_value"]) if len(with_edges) > 0 else None,
        "best_gate_without_edges": float(without_edges.loc[without_edges["ari"].idxmax(), "gate_value"]) if len(without_edges) > 0 else None,
        "optimal_gate_range": f"{optc_results['gate_value'].min():.2f}-{optc_results['gate_value'].max():.2f}"
    }
    
    # Bridge edge detailed analysis
    if len(with_edges) > 0:
        analysis["bridge_edge_analysis"] = {
            "avg_bridge_edge_pct": float(with_edges["bridge_edge_pct"].mean()),
            "max_bridge_edge_pct": float(with_edges["bridge_edge_pct"].max()),
            "correlation_with_ari": float(np.corrcoef(with_edges["bridge_edge_pct"], with_edges["ari"])[0, 1]),
            "unique_flows": int(with_edges["unique_bridge_flows"].max())
        }
        
        print(f"🔗 Bridge edge coverage: {with_edges['bridge_edge_pct'].mean():.1%} average")
        print(f"🔗 Max bridge edge coverage: {with_edges['bridge_edge_pct'].max():.1%}")
        print(f"🔗 Unique bridge flows: {with_edges['unique_bridge_flows'].max()}")
    
    return analysis

def create_visualizations(analysis: dict, output_dir: str = "experiments/results"):
    """Create visualizations for bridge edge analysis."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Load original data for plotting
    results = pd.read_csv("experiments/results/gate_tuning_results_v17.csv")
    optc_results = results[results["dataset"] == "OpTC"].copy()
    
    if len(optc_results) == 0:
        print("❌ No data for visualizations")
        return
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('OpTC Bridge Edge Analysis', fontsize=16, fontweight='bold')
    
    # 1. ARI comparison with/without bridge edges
    with_edges = optc_results[optc_results["bridge_edge_pct"] > 0]
    without_edges = optc_results[optc_results["bridge_edge_pct"] == 0]
    
    if len(with_edges) > 0 and len(without_edges) > 0:
        axes[0, 0].boxplot([without_edges["ari"], with_edges["ari"]], 
                          labels=['Without Bridge Edges', 'With Bridge Edges'])
        axes[0, 0].set_title('ARI Distribution: With vs Without Bridge Edges')
        axes[0, 0].set_ylabel('Adjusted Rand Index')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Bridge edge percentage vs ARI scatter plot
    if len(with_edges) > 0:
        axes[0, 1].scatter(with_edges["bridge_edge_pct"], with_edges["ari"], 
                          alpha=0.7, s=60, color='blue')
        axes[0, 1].set_xlabel('Bridge Edge Percentage')
        axes[0, 1].set_ylabel('ARI')
        axes[0, 1].set_title('Bridge Edge Coverage vs Clustering Performance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(with_edges["bridge_edge_pct"], with_edges["ari"], 1)
        p = np.poly1d(z)
        axes[0, 1].plot(with_edges["bridge_edge_pct"], p(with_edges["bridge_edge_pct"]), 
                       "r--", alpha=0.8, label=f'Trend (r={analysis["bridge_edge_analysis"]["correlation_with_ari"]:.3f})')
        axes[0, 1].legend()
    
    # 3. Gate value performance comparison
    if len(with_edges) > 0 and len(without_edges) > 0:
        gate_values = sorted(optc_results["gate_value"].unique())
        with_means = [with_edges[with_edges["gate_value"] == g]["ari"].mean() for g in gate_values]
        without_means = [without_edges[without_edges["gate_value"] == g]["ari"].mean() for g in gate_values]
        
        axes[1, 0].plot(gate_values, without_means, 'o-', label='Without Bridge Edges', linewidth=2)
        axes[1, 0].plot(gate_values, with_means, 's-', label='With Bridge Edges', linewidth=2)
        axes[1, 0].set_xlabel('Confidence Gate Value')
        axes[1, 0].set_ylabel('ARI')
        axes[1, 0].set_title('Gate Tuning: Performance Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Campaign distribution (if campaign data available)
    if "CampaignId" in optc_results.columns:
        campaign_counts = optc_results[optc_results["CampaignId"] != "Unknown"]["CampaignId"].value_counts()
        if len(campaign_counts) > 0:
            axes[1, 1].bar(range(len(campaign_counts)), campaign_counts.values)
            axes[1, 1].set_xticks(range(len(campaign_counts)))
            axes[1, 1].set_xticklabels(campaign_counts.index, rotation=45, ha='right')
            axes[1, 1].set_title('APT Campaign Distribution')
            axes[1, 1].set_ylabel('Number of Records')
            axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / "optc_bridge_edge_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Visualizations saved to: {output_path}/optc_bridge_edge_analysis.png")

def main():
    """Main analysis function."""
    
    print("🔍 Analyzing OpTC bridge edge impact on clustering performance...")
    
    # Run analysis
    analysis = analyze_optc_bridge_edges()
    
    if analysis is None:
        print("❌ Analysis failed - no OpTC data found")
        return
    
    # Save analysis results
    output_path = Path("experiments/results")
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "optc_bridge_edge_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print(f"📋 Analysis results saved to: {output_path}/optc_bridge_edge_analysis.json")
    
    # Create visualizations
    create_visualizations(analysis, str(output_path))
    
    # Print summary
    print("\n" + "="*60)
    print("📊 OPTC BRIDGE EDGE ANALYSIS SUMMARY")
    print("="*60)
    
    if "statistical_tests" in analysis:
        ari_test = analysis["statistical_tests"]["ari_ttest"]
        print(f"🎯 ARI Improvement: {analysis['ari_comparison']['improvement']:.4f}")
        print(f"📈 Statistical Significance: p = {ari_test['p_value']:.4f} ({'✅ SIGNIFICANT' if ari_test['significant'] else '❌ NOT SIGNIFICANT'})")
        print(f"📏 Effect Size: {ari_test['effect_size']:.3f}")
    
    if "bridge_edge_analysis" in analysis:
        bridge_data = analysis["bridge_edge_analysis"]
        print(f"🔗 Bridge Edge Coverage: {bridge_data['avg_bridge_edge_pct']:.1%}")
        print(f"🔗 Max Coverage: {bridge_data['max_bridge_edge_pct']:.1%}")
        print(f"🔗 Unique Bridge Flows: {bridge_data['unique_flows']:,}")
    
    print(f"📊 Total Records Analyzed: {analysis['dataset_info']['total_results']:,}")
    print("="*60)

if __name__ == "__main__":
    main()
