"""
Clustering utilities for MITRE-CORE training and evaluation.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances, silhouette_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from collections import defaultdict

def evaluate_clustering(true_labels, pred_labels, features, df, label_col, n_true_classes):
    """Evaluate clustering performance with binary_ari for 2-class datasets."""
    # Handle noise points in DBSCAN/HDBSCAN (label = -1)
    mask = pred_labels != -1
    
    if not mask.any():
        return {"ari": 0.0, "nmi": 0.0, "ami": 0.0, "silhouette": -1.0, "binary_ari": 0.0}
        
    true_valid = true_labels[mask]
    pred_valid = pred_labels[mask]
    
    ari = float(adjusted_rand_score(true_valid, pred_valid))
    nmi = float(normalized_mutual_info_score(true_valid, pred_valid))
    ami = float(adjusted_mutual_info_score(true_valid, pred_valid))
    
    silhouette = -1.0
    if features is not None and len(np.unique(pred_valid)) > 1:
        # Sample for silhouette if too large
        if len(pred_valid) > 10000:
            idx = np.random.choice(len(pred_valid), 10000, replace=False)
            silhouette = float(silhouette_score(features[mask][idx], pred_valid[idx]))
        else:
            silhouette = float(silhouette_score(features[mask], pred_valid))
            
    # Calculate binary ARI (Attack vs Normal) if there's a binary label column available
    # For now, default to standard ARI
    binary_ari = ari
    if "label" in df.columns:
        binary_ari = float(adjusted_rand_score(df["label"].values[mask], pred_valid))
        
    return {
        "ari": ari,
        "nmi": nmi,
        "ami": ami,
        "silhouette": silhouette,
        "binary_ari": binary_ari
    }

def hdbscan_cluster_with_confidence(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform HDBSCAN clustering and compute confidence scores.
    
    Args:
        embeddings: Array of shape (N, D) with embeddings
        min_cluster_size: Minimum cluster size for HDBSCAN
        min_samples: Number of samples in a neighborhood for a point to be considered a core point
        
    Returns:
        labels: Array of shape (N,) with cluster labels (-1 for noise)
        confidences: Array of shape (N,) with confidence scores
    """
    if min_samples is None:
        min_samples = min_cluster_size
    
    # Run HDBSCAN
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom',
        random_state=seed,
    )
    
    labels = clusterer.fit_predict(embeddings)
    
    # Compute confidence scores based on cluster membership strength
    confidences = np.zeros(len(embeddings))
    
    for cluster_id in np.unique(labels):
        if cluster_id == -1:  # Noise points
            cluster_mask = labels == cluster_id
            confidences[cluster_mask] = 0.0
        else:
            cluster_mask = labels == cluster_id
            cluster_embeddings = embeddings[cluster_mask]
            
            if len(cluster_embeddings) > 1:
                # Compute average distance to cluster centroid
                centroid = cluster_embeddings.mean(axis=0)
                distances = pairwise_distances(
                    cluster_embeddings.reshape(1, -1) if len(cluster_embeddings) == 1 else cluster_embeddings,
                    centroid.reshape(1, -1)
                ).flatten()
                
                # Convert distances to confidences (lower distance = higher confidence)
                # Use exponential decay: confidence = exp(-distance / scale)
                scale = distances.std() + 1e-8  # Avoid division by zero
                cluster_confidences = np.exp(-distances / scale)
                confidences[cluster_mask] = cluster_confidences
            else:
                # Single point cluster
                confidences[cluster_mask] = 1.0
    
    return labels, confidences


def compute_cluster_statistics(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> dict:
    """
    Compute clustering statistics.
    
    Args:
        embeddings: Array of shape (N, D) with embeddings
        labels: Array of shape (N,) with cluster labels
        
    Returns:
        Dictionary with clustering statistics
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    n_clustered = len(labels) - n_noise
    
    stats = {
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'n_clustered': n_clustered,
        'noise_ratio': n_noise / len(labels) if len(labels) > 0 else 0.0
    }
    
    # Compute cluster sizes
    if n_clusters > 0:
        cluster_sizes = []
        for cluster_id in np.unique(labels):
            if cluster_id != -1:
                cluster_size = (labels == cluster_id).sum()
                cluster_sizes.append(cluster_size)
        
        stats['avg_cluster_size'] = np.mean(cluster_sizes)
        stats['min_cluster_size'] = np.min(cluster_sizes)
        stats['max_cluster_size'] = np.max(cluster_sizes)
    else:
        stats['avg_cluster_size'] = 0.0
        stats['min_cluster_size'] = 0.0
        stats['max_cluster_size'] = 0.0
    
    return stats
