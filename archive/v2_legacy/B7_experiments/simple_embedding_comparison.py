#!/usr/bin/env python3
"""
Simple Embedding Comparison Script
==================================

Quick comparison of SupCon vs baseline embeddings using k-NN accuracy.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from hgnn.hgnn_correlation import HGNNCorrelationEngine
from training.train_on_datasets import PublicDatasetGraphConverter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_embeddings(dataset_path: str, checkpoint_path: str, method_name: str):
    """Evaluate embedding quality using k-NN accuracy."""
    logger.info(f"Evaluating {method_name} embeddings...")
    
    # Load dataset
    df = pd.read_parquet(dataset_path) if dataset_path.endswith('.parquet') else pd.read_csv(dataset_path)
    
    # Sample if dataset is too large
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled to {len(df)} records")
    
    # Convert to graph
    converter = PublicDatasetGraphConverter()
    graph = converter.convert(df)
    
    # Load model
    hgnn = HGNNCorrelationEngine(
        model_path=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        pure_unsupervised=True
    )
    
    # Extract embeddings
    with torch.no_grad():
        embeddings, _ = hgnn.correlate(graph)
    
    # Get labels
    labels = df['campaign_id'].values if 'campaign_id' in df.columns else df['label'].values
    
    # Evaluate k-NN accuracy
    embeddings_np = embeddings.cpu().numpy()
    
    try:
        knn = KNeighborsClassifier(n_neighbors=5)
        knn_scores = cross_val_score(knn, embeddings_np, labels, cv=min(5, len(set(labels))), scoring='accuracy')
        
        accuracy = knn_scores.mean()
        std = knn_scores.std()
        
        logger.info(f"{method_name} k-NN accuracy: {accuracy:.3f} ± {std:.3f}")
        
        return {
            'method': method_name,
            'dataset': os.path.basename(dataset_path),
            'n_samples': len(embeddings_np),
            'n_features': embeddings_np.shape[1],
            'n_classes': len(np.unique(labels)),
            'knn_accuracy': accuracy,
            'knn_accuracy_std': std
        }
        
    except Exception as e:
        logger.error(f"k-NN evaluation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Simple embedding comparison")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Dataset path")
    parser.add_argument("--supcon", type=str, help="SupCon checkpoint path")
    parser.add_argument("--baseline", type=str, help="Baseline checkpoint path")
    parser.add_argument("--output", type=str, default="experiments/results/simple_embedding_comparison.csv",
                       help="Output file")
    
    args = parser.parse_args()
    
    results = []
    
    # Evaluate SupCon if provided
    if args.supcon and os.path.exists(args.supcon):
        result = evaluate_embeddings(args.dataset, args.supcon, "SupCon")
        if result:
            results.append(result)
    
    # Evaluate baseline if provided
    if args.baseline and os.path.exists(args.baseline):
        result = evaluate_embeddings(args.dataset, args.baseline, "Baseline")
        if result:
            results.append(result)
    
    # Save results
    if results:
        df_results = pd.DataFrame(results)
        
        # Append to existing results or create new file
        if os.path.exists(args.output):
            existing_df = pd.read_csv(args.output)
            df_results = pd.concat([existing_df, df_results], ignore_index=True)
        
        df_results.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
        
        # Print comparison
        print("\n=== Embedding Comparison ===")
        for result in results:
            print(f"{result['method']}: {result['knn_accuracy']:.3f} ± {result['knn_accuracy_std']:.3f}")
    else:
        logger.error("No results to save")


if __name__ == "__main__":
    main()
