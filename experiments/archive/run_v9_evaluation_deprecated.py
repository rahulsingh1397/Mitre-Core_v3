"""
Network v9 Evaluation Script
Evaluates network_v9 architecture with re-tuned HDBSCAN parameters for spread embedding geometry.
Reuses run_gate_tuning.py infrastructure but overrides configurations for v9-specific needs.
"""

import sys
import os
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import core functions from run_gate_tuning.py
from experiments.run_gate_tuning import run_sweep, load_dataset, encode_labels, DATASET_CONFIG
from utils.seed_control import set_seed

# Network v9 specific dataset configurations
DATASET_CONFIG_V9 = {
    "UNSW-NB15": {
        "path": "datasets/unsw_nb15/mitre_format.csv",
        "label_col": "campaign_id",
        "checkpoint_override": "hgnn_checkpoints/network_v9_v2_fast/network_it_best.pt",
        # Re-tuned for cosine_sim=0.26 geometry (was tuned for cosine_sim=0.99)
        "hdbscan_min_cluster_size": 10,          # was 30 - smaller for spread embeddings
        "hdbscan_pca_components": 16,
        "hdbscan_cluster_selection_epsilon": 0.0, # was 0.05 - no merging, let HDBSCAN decide
        "use_umap": False,                         # was True - embeddings already spread, UMAP unnecessary
        "use_geometric_confidence": True,
        "hdbscan_auto_tune": True,
        "sample_size": 2000,
        "stratified_sample": True,
        "chunk_inference_size": 1000,
    },
    "NSL-KDD": {
        "path": "datasets/nsl_kdd/mitre_format.csv",
        "label_col": "tactic",
        "checkpoint_override": "hgnn_checkpoints/network_v9_v2_fast/network_it_best.pt",  # zero-shot transfer
        # NSL-KDD has real IP co-occurrence; keep smaller cluster sizes
        "hdbscan_min_cluster_size": 5,
        "hdbscan_pca_components": 16,
        "hdbscan_cluster_selection_epsilon": 0.0,
        "use_geometric_confidence": True,
        "sample_size": 10000,
        "stratified_sample": True,
    },
    "TON_IoT": {
        "path": "datasets/TON_IoT/mitre_format.parquet",
        "label_col": "campaign_id",
        "checkpoint_override": "hgnn_checkpoints/network_v9_v2_fast/network_it_best.pt",
        "hdbscan_min_cluster_size": 10,   # was 15
        "hdbscan_pca_components": 16,
        "hdbscan_cluster_selection_epsilon": 0.0,   # was 0.1
        "use_umap": False,                           # was True
        "use_geometric_confidence": True,
        "hdbscan_auto_tune": True,
        "sample_size": 10000,
        "stratified_sample": True,
    },
    "Attack_Techniques": {
        "path": "datasets/attack_data_processed/attack_techniques_mitre_format.parquet",
        "label_col": "campaign_id",
        "checkpoint_override": "hgnn_checkpoints/network_v9_v2_fast/network_it_best.pt",
        "hdbscan_min_cluster_size": 5,
        "hdbscan_pca_components": 16,
        "hdbscan_auto_tune": True,
        "use_geometric_confidence": True,
        "sample_size": 5000,
        "stratified_sample": True,
    },
    "OpTC": {
        "path": "datasets/DARPA_OpTC/processed_optc_full.csv",
        "label_col": "CampaignId",
        "checkpoint_override": "hgnn_checkpoints/network_v9_v2_fast/network_it_best.pt",
        "hdbscan_min_cluster_size": 20,   # was 50
        "hdbscan_pca_components": 8,
        "use_geometric_confidence": True,
        "sample_size": 10000,
        "stratified_sample": True,
        "chunk_inference_size": 1000,
    },
    "BETH": {
        "path": "datasets/BETH/mitre_format.parquet",
        "label_col": "campaign_id",
        "checkpoint_override": "hgnn_checkpoints/network_v9_v2_fast/network_it_best.pt",
        "hdbscan_min_cluster_size": 20,   # was 50
        "hdbscan_pca_components": 32,
        "use_geometric_confidence": True,
        "hdbscan_auto_tune": True,
        "sample_size": 10000,
        "stratified_sample": True,
    },
}

# Epsilon sweep values for UNSW-NB15 (highly sensitive when embeddings are spread)
EPSILON_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def run_epsilon_sweep(dataset_name: str, output_path: str):
    """Run epsilon sweep for a specific dataset."""
    print(f"Running epsilon sweep for {dataset_name}...")
    
    # Temporarily modify DATASET_CONFIG for epsilon sweep
    original_config = DATASET_CONFIG.get(dataset_name, {}).copy()
    results = []
    
    for epsilon in EPSILON_VALUES:
        print(f"Testing epsilon={epsilon}")
        
        # Update global DATASET_CONFIG with current epsilon
        DATASET_CONFIG[dataset_name] = DATASET_CONFIG_V9[dataset_name].copy()
        DATASET_CONFIG[dataset_name]["hdbscan_cluster_selection_epsilon"] = epsilon
        
        # Run single configuration
        try:
            sweep_output = output_path.replace('.csv', f'_eps{epsilon}.csv')
            run_sweep(
                checkpoint_path=DATASET_CONFIG[dataset_name]["checkpoint_override"],
                output_path=sweep_output,
                datasets=[dataset_name]
            )
            
            # Read the results
            if os.path.exists(sweep_output):
                result_df = pd.read_csv(sweep_output)
                if len(result_df) > 0:
                    row = result_df.iloc[0].copy()
                    row["epsilon"] = epsilon
                    results.append(row)
                    
        except Exception as e:
            print(f"Error with epsilon={epsilon}: {e}")
            continue
    
    # Restore original config
    if original_config:
        DATASET_CONFIG[dataset_name] = original_config
    elif dataset_name in DATASET_CONFIG:
        del DATASET_CONFIG[dataset_name]
    
    # Combine results
    if results:
        sweep_results = pd.DataFrame(results)
        sweep_output = output_path.replace('.csv', '_epsilon_sweep.csv')
        sweep_results.to_csv(sweep_output, index=False)
        print(f"Epsilon sweep results saved to {sweep_output}")
        
        # Find best epsilon by ARI
        best_idx = sweep_results["ari"].idxmax()
        best_row = sweep_results.loc[best_idx]
        print(f"Best epsilon for {dataset_name}: {best_row['epsilon']} (ARI={best_row['ari']:.4f})")
        
        return sweep_results
    else:
        print(f"No successful results for {dataset_name}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate network_v9 with re-tuned HDBSCAN parameters")
    parser.add_argument("--datasets", nargs="+", default=["UNSW-NB15", "NSL-KDD"],
                       help="Datasets to evaluate")
    parser.add_argument("--output", default="experiments/results/network_v9_evaluation.csv",
                       help="Output file for results")
    parser.add_argument("--epsilon_sweep", action="store_true",
                       help="Run epsilon sweep for UNSW-NB15")
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")
    parser.add_argument('--seed', type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Network v9 Evaluation")
    print("=" * 60)
    print(f"Datasets: {args.datasets}")
    print(f"Output: {args.output}")
    print(f"Epsilon sweep: {args.epsilon_sweep}")
    print()
    
    if args.epsilon_sweep and "UNSW-NB15" in args.datasets:
        # Run epsilon sweep for UNSW-NB15
        unsw_results = run_epsilon_sweep("UNSW-NB15", args.output)
        
        # Also run NSL-KDD with default epsilon=0.0
        if "NSL-KDD" in args.datasets:
            print("\nRunning NSL-KDD evaluation...")
            try:
                # Temporarily update DATASET_CONFIG for NSL-KDD
                original_nsl_config = DATASET_CONFIG.get("NSL-KDD", {}).copy()
                DATASET_CONFIG["NSL-KDD"] = DATASET_CONFIG_V9["NSL-KDD"].copy()
                
                nsl_output = args.output.replace('.csv', '_nslkdd.csv')
                run_sweep(
                    checkpoint_path=DATASET_CONFIG["NSL-KDD"]["checkpoint_override"],
                    output_path=nsl_output,
                    datasets=["NSL-KDD"]
                )
                
                # Read results
                if os.path.exists(nsl_output):
                    nsl_results = pd.read_csv(nsl_output)
                    if len(nsl_results) > 0:
                        print(f"NSL-KDD zero-shot ARI: {nsl_results.iloc[0]['ari']:.4f}")
                        
                        # Check if we need domain-specific training
                        nsl_ari = nsl_results.iloc[0]['ari']
                        if nsl_ari < 0.20:
                            print("NSL-KDD zero-shot ARI < 0.20, domain-specific training recommended")
                            print("Run: python training/train_graph_mae_v9.py --datasets nsl_kdd --epochs 150")
                        else:
                            print("NSL-KDD zero-shot ARI >= 0.20, transfer generalizes adequately")
                
                # Restore original config
                if original_nsl_config:
                    DATASET_CONFIG["NSL-KDD"] = original_nsl_config
                elif "NSL-KDD" in DATASET_CONFIG:
                    del DATASET_CONFIG["NSL-KDD"]
                        
            except Exception as e:
                print(f"Error running NSL-KDD evaluation: {e}")
        
    else:
        # Run standard evaluation
        print("Running standard evaluation...")
        try:
            # Temporarily update DATASET_CONFIG for v9 settings
            original_configs = {}
            for dataset_name in args.datasets:
                if dataset_name in DATASET_CONFIG_V9:
                    original_configs[dataset_name] = DATASET_CONFIG.get(dataset_name, {}).copy()
                    DATASET_CONFIG[dataset_name] = DATASET_CONFIG_V9[dataset_name].copy()
            
            run_sweep(
                checkpoint_path="hgnn_checkpoints/network_v9_v2_fast/network_it_best.pt",
                output_path=args.output,
                datasets=args.datasets
            )
            
            # Read and display results
            if os.path.exists(args.output):
                results = pd.read_csv(args.output)
                if len(results) > 0:
                    print("\nResults Summary:")
                    print(results[['dataset', 'ari', 'nmi', 'silhouette', 'clusters_found', 'noise_ratio']].to_string())
                    
                    # Check NSL-KDD zero-shot performance
                    nsl_row = results[results['dataset'] == 'NSL-KDD']
                    if len(nsl_row) > 0:
                        nsl_ari = nsl_row.iloc[0]['ari']
                        if nsl_ari < 0.20:
                            print(f"\nNSL-KDD zero-shot ARI ({nsl_ari:.4f}) < 0.20, domain-specific training recommended")
                        else:
                            print(f"\nNSL-KDD zero-shot ARI ({nsl_ari:.4f}) >= 0.20, transfer generalizes adequately")
            
            # Restore original configs
            for dataset_name, original_config in original_configs.items():
                if original_config:
                    DATASET_CONFIG[dataset_name] = original_config
                elif dataset_name in DATASET_CONFIG:
                    del DATASET_CONFIG[dataset_name]
                        
        except Exception as e:
            print(f"Error running evaluation: {e}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
