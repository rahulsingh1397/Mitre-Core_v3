"""
Metrics utilities for MITRE-CORE evaluation.
"""

import numpy as np
from typing import Optional
from sklearn.metrics import adjusted_rand_score as sklearn_ari
from sklearn.metrics import normalized_mutual_info_score as sklearn_nmi


def adjusted_rand_score(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    binary: bool = False
) -> float:
    """
    Compute Adjusted Rand Index (ARI).
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        binary: Whether to compute binary ARI for 2-class datasets
        
    Returns:
        ARI score
    """
    if binary and len(np.unique(labels_true)) == 2:
        # For binary datasets, map predicted clusters to majority class
        return binary_ari(labels_true, labels_pred)
    
    return sklearn_ari(labels_true, labels_pred)


def normalized_mutual_info_score(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    binary: bool = False
) -> float:
    """
    Compute Normalized Mutual Information (NMI).
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        binary: Whether to compute binary NMI for 2-class datasets
        
    Returns:
        NMI score
    """
    if binary and len(np.unique(labels_true)) == 2:
        # For binary datasets, map predicted clusters to majority class
        return binary_nmi(labels_true, labels_pred)
    
    return sklearn_nmi(labels_true, labels_pred)


def binary_ari(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute binary ARI for 2-class datasets.
    Maps predicted clusters to majority ground-truth class.
    """
    # Get unique predicted clusters (excluding noise if present)
    unique_pred = np.unique(labels_pred[labels_pred != -1])
    
    if len(unique_pred) == 0:
        return 0.0
    
    # Map each predicted cluster to majority true class
    mapped_pred = np.full_like(labels_pred, -1)
    
    for pred_cluster in unique_pred:
        mask = labels_pred == pred_cluster
        if mask.sum() == 0:
            continue
            
        # Find majority true class for this predicted cluster
        true_classes = labels_true[mask]
        majority_class = np.bincount(true_classes).argmax()
        mapped_pred[mask] = majority_class
    
    # Compute ARI on mapped predictions
    valid_mask = mapped_pred != -1
    if valid_mask.sum() < 2:
        return 0.0
    
    return sklearn_ari(labels_true[valid_mask], mapped_pred[valid_mask])


def binary_nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute binary NMI for 2-class datasets.
    Maps predicted clusters to majority ground-truth class.
    """
    # Get unique predicted clusters (excluding noise if present)
    unique_pred = np.unique(labels_pred[labels_pred != -1])
    
    if len(unique_pred) == 0:
        return 0.0
    
    # Map each predicted cluster to majority true class
    mapped_pred = np.full_like(labels_pred, -1)
    
    for pred_cluster in unique_pred:
        mask = labels_pred == pred_cluster
        if mask.sum() == 0:
            continue
            
        # Find majority true class for this predicted cluster
        true_classes = labels_true[mask]
        majority_class = np.bincount(true_classes).argmax()
        mapped_pred[mask] = majority_class
    
    # Compute NMI on mapped predictions
    valid_mask = mapped_pred != -1
    if valid_mask.sum() < 2:
        return 0.0
    
    return sklearn_nmi(labels_true[valid_mask], mapped_pred[valid_mask])


def silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray
) -> float:
    """
    Compute silhouette score for clustering evaluation.
    
    Args:
        embeddings: Array of shape (N, D) with embeddings
        labels: Array of shape (N,) with cluster labels
        
    Returns:
        Silhouette score
    """
    from sklearn.metrics import silhouette_score as sklearn_silhouette
    
    # Filter out noise points
    valid_mask = labels != -1
    if valid_mask.sum() < 2 or len(np.unique(labels[valid_mask])) < 2:
        return 0.0
    
    return sklearn_silhouette(embeddings[valid_mask], labels[valid_mask])
