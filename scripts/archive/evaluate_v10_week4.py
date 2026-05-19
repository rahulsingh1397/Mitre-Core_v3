#!/usr/bin/env python3
"""
Evaluate v10 Week 4 stabilized models and generate gate tuning results.
"""

import sys
import time
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from hgnn.hgnn_correlation import MITREHeteroGNN
from training.train_on_datasets import PublicDatasetGraphConverter
from utils.clustering import hdbscan_cluster_with_confidence
from utils.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_checkpoint(checkpoint_path: str, dataset_name: str, gate_values: List[float]) -> pd.DataFrame:
    """
    Evaluate a single checkpoint across different gate values.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        dataset_name: Name of the dataset to evaluate on
        gate_values: List of gate values to test
        
    Returns:
        DataFrame with evaluation results
    """
    logger.info(f"Evaluating {checkpoint_path} on {dataset_name}")
    
    # Load dataset
    if dataset_name == 'TON_IoT':
        df = pd.read_parquet('datasets/TON_IoT/mitre_format.parquet')
    elif dataset_name == 'UNSW-NB15':
        df = pd.read_csv('datasets/unsw_nb15/mitre_format.csv')
    else:
        logger.warning(f"Unknown dataset: {dataset_name}")
        return pd.DataFrame()
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get vocab sizes from checkpoint if available
    vocab_sizes = checkpoint.get('vocab_sizes', {})
    
    # Initialize model with correct architecture
    hgnn = MITREHeteroGNN(
        alert_feature_dim=6,
        hidden_dim=checkpoint.get('hidden_dim', 128),
        num_layers=1,
        num_clusters=checkpoint.get('num_clusters', 10),
        vocab_sizes=vocab_sizes,
    ).to(device)
    
    # Load with strict=False to handle architecture differences
    missing_keys, unexpected_keys = hgnn.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    hgnn.eval()
    
    # Convert to graph
    converter = PublicDatasetGraphConverter()
    graph = converter.convert(df)
    if graph is None:
        logger.error(f"Failed to convert {dataset_name} to graph")
        return pd.DataFrame()
    
    graph = graph.to(device)
    
    # Get embeddings
    with torch.no_grad():
        output = hgnn(graph)
        # Handle both single tensor and tuple returns
        if isinstance(output, tuple):
            embeddings = output[0].cpu().numpy() if 'alert' in output[1] else output[0].cpu().numpy()
        else:
            embeddings = output['alert'].cpu().numpy() if 'alert' in output else output.cpu().numpy()
    
    results = []
    
    for gate in gate_values:
        logger.info(f"  Testing gate={gate}")
        
        start_time = time.time()
        
        try:
            # Cluster embeddings
            cluster_labels, confidences = hdbscan_cluster_with_confidence(
                embeddings,
                min_cluster_size=10
            )
            
            # Apply gate threshold
            high_conf_mask = confidences >= gate
            final_labels = np.full(len(cluster_labels), -1, dtype=int)
            final_labels[high_conf_mask] = cluster_labels[high_conf_mask]
            
            # Compute metrics
            if 'campaign_id' in df.columns:
                true_labels = df['campaign_id'].values
                
                # Filter out noise (-1) for ARI/NMI
                valid_mask = (final_labels != -1)
                if valid_mask.sum() > 0:
                    ari = adjusted_rand_score(true_labels[valid_mask], final_labels[valid_mask])
                    nmi = normalized_mutual_info_score(true_labels[valid_mask], final_labels[valid_mask])
                else:
                    ari = np.nan
                    nmi = np.nan
                
                # Binary ARI (attack vs benign)
                if 'attack' in df.columns:
                    binary_true = (df['attack'] == 1).values
                    binary_pred = (final_labels >= 0).values  # Any cluster = attack
                    binary_ari = adjusted_rand_score(binary_true[valid_mask], binary_pred[valid_mask])
                else:
                    binary_ari = np.nan
            else:
                ari = np.nan
                nmi = np.nan
                binary_ari = np.nan
            
            # Silhouette score
            try:
                silhouette = silhouette_score(embeddings[valid_mask], final_labels[valid_mask])
            except:
                silhouette = np.nan
            
            eval_time = time.time() - start_time
            
            results.append({
                'checkpoint_path': checkpoint_path,
                'dataset': dataset_name,
                'gate': gate,
                'ari': ari,
                'nmi': nmi,
                'binary_ari': binary_ari,
                'silhouette': silhouette,
                'eval_time_seconds': eval_time,
                'n_clusters': len(np.unique(final_labels[final_labels != -1])),
                'noise_ratio': (final_labels == -1).mean()
            })
            
        except Exception as e:
            logger.error(f"    Error evaluating gate={gate}: {e}")
            results.append({
                'checkpoint_path': checkpoint_path,
                'dataset': dataset_name,
                'gate': gate,
                'ari': np.nan,
                'nmi': np.nan,
                'binary_ari': np.nan,
                'silhouette': np.nan,
                'eval_time_seconds': time.time() - start_time,
                'n_clusters': np.nan,
                'noise_ratio': np.nan
            })
    
    return pd.DataFrame(results)

def main():
    """Main evaluation function."""
    # Load v10_week4_stabilized.csv
    results_df = pd.read_csv('experiments/results/v10_week4_stabilized.csv')
    
    # Gate values to test
    gate_values = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    all_results = []
    
    for _, row in results_df.iterrows():
        checkpoint_path = row['checkpoint_path']
        dataset_name = row['dataset']
        
        # Skip SQTK_SIEM (dataset not found)
        if dataset_name == 'SQTK_SIEM':
            continue
        
        # Evaluate checkpoint
        eval_results = evaluate_checkpoint(checkpoint_path, dataset_name, gate_values)
        all_results.append(eval_results)
    
    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Save results
    output_path = 'experiments/results/v10_week4_gate_evaluation.csv'
    combined_results.to_csv(output_path, index=False)
    logger.info(f"Gate evaluation complete. Results saved to {output_path}")
    
    # Print summary
    logger.info("\n=== BEST GATE VALUES ===")
    for dataset in combined_results['dataset'].unique():
        dataset_results = combined_results[combined_results['dataset'] == dataset]
        best_row = dataset_results.loc[dataset_results['ari'].idxmax()]
        logger.info(f"{dataset}: gate={best_row['gate']:.2f}, ARI={best_row['ari']:.3f}")
    
    return combined_results

if __name__ == '__main__':
    main()
