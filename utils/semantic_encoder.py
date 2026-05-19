"""
Lightweight semantic encoder for alert_type / RequestURL fields (Phase 1.3)

Uses MiniLM-L6-v2 (384-dim) sentence embeddings cached offline to avoid
runtime dependencies and latency. For SIEM datasets where src_bytes/dst_bytes
are zeroed, this provides real discriminative signal.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

class CachedSemanticEncoder:
    """
    Precompute and cache MiniLM embeddings for categorical text fields.
    """
    def __init__(self, cache_dir: str = "cache/semantic_embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.minilm_dim = 384
        self.pca_dim = 32  # After PCA reduction
        self.pca = None
        self._load_pca()

    def _load_pca(self):
        pca_path = self.cache_dir / "pca.pkl"
        if pca_path.exists():
            import joblib
            self.pca = joblib.load(pca_path)
            logger.info(f"Loaded PCA from {pca_path}")
        else:
            logger.info("PCA not found; will fit on first batch")

    def _fit_pca(self, embeddings: np.ndarray):
        """Fit PCA on first batch and save."""
        from sklearn.decomposition import PCA
        self.pca = PCA(n_components=self.pca_dim, random_state=42)
        reduced = self.pca.fit_transform(embeddings)
        import joblib
        joblib.dump(self.pca, self.cache_dir / "pca.pkl")
        logger.info(f"Fitted PCA: {embeddings.shape[1]} -> {self.pca_dim} dims, explained={self.pca.explained_variance_ratio_.sum():.3f}")
        return reduced

    def _get_cache_path(self, texts: List[str]) -> Path:
        """Deterministic cache path based on text hash."""
        hasher = hashlib.sha256()
        for t in sorted(texts):
            hasher.update(t.encode())
        return self.cache_dir / f"minilm_{hasher.hexdigest()[:16]}.npz"

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to reduced PCA embeddings."""
        cache_path = self._get_cache_path(texts)
        if cache_path.exists():
            data = np.load(cache_path)
            logger.info(f"Loaded {len(texts)} embeddings from cache")
            return data["embeddings"]

        # Compute MiniLM embeddings (requires sentence-transformers)
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise

        logger.info(f"Computing MiniLM embeddings for {len(texts)} texts...")
        embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

        # Fit PCA on first batch if needed
        if self.pca is None:
            reduced = self._fit_pca(embeddings)
        else:
            reduced = self.pca.transform(embeddings)

        # Save to cache
        np.savez_compressed(cache_path, embeddings=reduced)
        logger.info(f"Cached reduced embeddings to {cache_path}")
        return reduced

    def encode_dataframe_column(self, df: pd.DataFrame, column: str) -> np.ndarray:
        """Encode a pandas Series column."""
        texts = df[column].fillna("").astype(str).tolist()
        return self.encode(texts)

def add_semantic_features(df: pd.DataFrame, cache_dir: str = "cache/semantic_embeddings") -> pd.DataFrame:
    """
    Add semantic embeddings for alert_type and RequestURL (if present).
    Returns a copy with added columns: alert_type_sem, requesturl_sem (if applicable).
    """
    df = df.copy()
    encoder = CachedSemanticEncoder(cache_dir)

    # Encode alert_type
    if "alert_type" in df.columns:
        alert_embeddings = encoder.encode_dataframe_column(df, "alert_type")
        df[[f"alert_type_sem_{i}" for i in range(alert_embeddings.shape[1])]] = alert_embeddings
        logger.info(f"Added {alert_embeddings.shape[1]} alert_type semantic features")

    # Encode RequestURL if present (SIEM datasets often have URLs)
    if "RequestURL" in df.columns:
        url_embeddings = encoder.encode_dataframe_column(df, "RequestURL")
        df[[f"requesturl_sem_{i}" for i in range(url_embeddings.shape[1])]] = url_embeddings
        logger.info(f"Added {url_embeddings.shape[1]} RequestURL semantic features")

    return df
