#!/usr/bin/env python3
"""
NSL-KDD Feature-Only Baseline Script

Evaluates whether graph structure adds value for NSL-KDD dataset by comparing
HGNN performance against feature-only baselines (K-Means, GMM, GBM).

NSL-KDD has zero shared entities (disconnected graph), so HGNN reduces to MLP.
This script provides an honest comparator to document alongside ARI=0.722.

Usage:
    python experiments/run_feature_baseline.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import LabelEncoder

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.dataset_registry import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_nsl_kdd_data() -> tuple[pd.DataFrame, np.ndarray]:
    """Load NSL-KDD dataset and extract features with ground truth labels."""
    
    # Load NSL-KDD dataset
    logger.info("Loading NSL-KDD dataset...")
    df = load_dataset("nsl_kdd")
    
    # Extract the 6 raw features used by HGNN
    feature_columns = ['tactic', 'alert_type', 'hour', 'day_of_week', 'protocol', 'service']
    
    # Check if columns exist
    available_features = [col for col in feature_columns if col in df.columns]
    missing_features = [col for col in feature_columns if col not in df.columns]
    
    if missing_features:
        logger.warning(f"Missing features: {missing_features}")
        logger.info(f"Using available features: {available_features}")
    
    # Extract features
    X = df[available_features].copy()
    
    # Handle categorical features
    categorical_features = ['tactic', 'alert_type', 'protocol', 'service']
    for col in categorical_features:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Get ground truth labels (use 'tactic' as proxy for attack families)
    if 'tactic' in df.columns:
        y = df['tactic'].fillna('unknown').astype(str).values
    else:
        logger.error("No ground truth labels found in NSL-KDD dataset")
        raise ValueError("Ground truth labels required for evaluation")
    
    logger.info(f"Loaded {len(X)} samples with {len(available_features)} features")
    logger.info(f"Ground truth classes: {len(np.unique(y))}")
    
    return X, y


def evaluate_kmeans(X: np.ndarray, y: np.ndarray, n_clusters: int = 4) -> float:
    """Evaluate K-Means clustering on raw features."""
    logger.info(f"Running K-Means with k={n_clusters}...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(X)
    
    ari = adjusted_rand_score(y, pred_labels)
    logger.info(f"K-Means ARI: {ari:.4f}")
    
    return ari


def evaluate_gmm(X: np.ndarray, y: np.ndarray, n_components: int = 4) -> float:
    """Evaluate Gaussian Mixture Model on raw features."""
    logger.info(f"Running GMM with k={n_components}...")
    
    gmm = GaussianMixture(n_components=n_components, random_state=42, n_init=10)
    pred_labels = gmm.fit_predict(X)
    
    ari = adjusted_rand_score(y, pred_labels)
    logger.info(f"GMM ARI: {ari:.4f}")
    
    return ari


def evaluate_supervised_gbm(X: np.ndarray, y: np.ndarray) -> float:
    """Evaluate supervised GBM as upper bound (uses labels during training)."""
    logger.info("Running supervised GBM (upper bound)...")
    
    # Encode labels for supervised learning
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    gbm = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    # Train on all data (upper bound scenario)
    gbm.fit(X, y_encoded)
    pred_labels = gbm.predict(X)
    
    # Convert back to original labels for ARI calculation
    pred_labels_original = le.inverse_transform(pred_labels)
    ari = adjusted_rand_score(y, pred_labels_original)
    
    logger.info(f"Supervised GBM ARI: {ari:.4f}")
    
    return ari


def main():
    """Run feature-only baseline evaluation for NSL-KDD."""
    logger.info("Starting NSL-KDD Feature-Only Baseline Evaluation")
    
    try:
        # Load data
        X, y = load_nsl_kdd_data()
        
        # Determine number of clusters (use number of unique attack families)
        n_clusters = len(np.unique(y))
        logger.info(f"Using {n_clusters} clusters based on unique attack families")
        
        # Run baseline methods
        results = {}
        
        # Unsupervised methods
        results['kmeans_ari'] = evaluate_kmeans(X.values, y, n_clusters)
        results['gmm_ari'] = evaluate_gmm(X.values, y, n_clusters)
        
        # Supervised upper bound
        results['supervised_gbm_ari'] = evaluate_supervised_gbm(X.values, y)
        
        # Reference HGNN performance (from memory/experiments)
        results['hgnn_ari'] = 0.722  # Reference value from MITRE-CORE experiments
        
        # Add metadata
        results['dataset'] = 'nsl_kdd'
        results['n_samples'] = len(X)
        results['n_features'] = X.shape[1]
        results['n_clusters'] = n_clusters
        results['feature_columns'] = list(X.columns)
        
        # Save results
        output_path = Path("experiments/results/nsl_kdd_feature_baseline.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        
        # Print summary
        logger.info("\n=== NSL-KDD Feature Baseline Results ===")
        logger.info(f"K-Means ARI:        {results['kmeans_ari']:.4f}")
        logger.info(f"GMM ARI:             {results['gmm_ari']:.4f}")
        logger.info(f"Supervised GBM ARI:  {results['supervised_gbm_ari']:.4f}")
        logger.info(f"HGNN ARI (ref):      {results['hgnn_ari']:.4f}")
        
        # Analysis
        best_unsupervised = max(results['kmeans_ari'], results['gmm_ari'])
        if best_unsupervised >= results['hgnn_ari']:
            logger.info("\nCONCLUSION: Feature-only methods match or exceed HGNN performance.")
            logger.info("Graph structure adds no value for NSL-KDD (disconnected graph).")
        else:
            logger.info("\nCONCLUSION: HGNN outperforms feature-only methods.")
            logger.info("Graph structure provides genuine benefit even on disconnected graph.")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in feature baseline evaluation: {e}")
        raise


if __name__ == "__main__":
    main()
