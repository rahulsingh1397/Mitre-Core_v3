from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, f1_score, silhouette_score


def purity_score(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    if y_true.size == 0:
        return 0.0
    total = 0
    for cluster in np.unique(y_pred):
        members = y_true[y_pred == cluster]
        if members.size == 0:
            continue
        total += Counter(members.tolist()).most_common(1)[0][1]
    return float(total / len(y_true))


def binary_ari_score(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    if len(np.unique(y_true)) != 2:
        return float("nan")
    cluster_majority = {}
    for cluster in np.unique(y_pred):
        members = y_true[y_pred == cluster]
        if members.size == 0:
            continue
        cluster_majority[cluster] = Counter(members.tolist()).most_common(1)[0][0]
    mapped = np.asarray([cluster_majority.get(c, y_true[0]) for c in y_pred])
    return float(adjusted_rand_score(y_true, mapped))


def attack_f1_score(y_true: Iterable, y_pred: Iterable, benign_label: str | int = 0) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    y_true_binary = (y_true != benign_label).astype(int)
    cluster_majority = {}
    for cluster in np.unique(y_pred):
        members = y_true_binary[y_pred == cluster]
        if members.size == 0:
            continue
        cluster_majority[cluster] = Counter(members.tolist()).most_common(1)[0][0]
    mapped = np.asarray([cluster_majority.get(c, 0) for c in y_pred])
    return float(f1_score(y_true_binary, mapped, zero_division=0))


def cluster_attribution_f1(y_true: Iterable, y_pred: Iterable) -> float:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    if y_true.size == 0:
        return 0.0
    truth_clusters = np.unique(y_true)
    matches = 0
    for truth_cluster in truth_clusters:
        mask = y_true == truth_cluster
        pred_members = y_pred[mask]
        if pred_members.size == 0:
            continue
        top_pred = Counter(pred_members.tolist()).most_common(1)[0][0]
        top_mask = y_pred == top_pred
        tp = int(np.sum(mask & top_mask))
        fp = int(np.sum(~mask & top_mask))
        fn = int(np.sum(mask & ~top_mask))
        denom = 2 * tp + fp + fn
        if denom > 0:
            matches += (2 * tp) / denom
    return float(matches / len(truth_clusters))


def compute_unsupervised_metrics(
    y_true: Iterable,
    y_pred: Iterable,
    embeddings: np.ndarray | None = None,
    benign_label: str | int = 0,
) -> dict:
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))
    metrics = {
        "ami": float(adjusted_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "binary_ari": binary_ari_score(y_true, y_pred),
        "purity": purity_score(y_true, y_pred),
        "attack_f1": attack_f1_score(y_true, y_pred, benign_label=benign_label),
        "cluster_attribution_f1": cluster_attribution_f1(y_true, y_pred),
        "silhouette_cosine": 0.0,
    }
    if embeddings is not None and len(np.unique(y_pred)) > 1 and len(y_pred) > len(np.unique(y_pred)):
        metrics["silhouette_cosine"] = float(silhouette_score(embeddings, y_pred, metric="cosine"))
    return metrics
