from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from hgnn.hgnn_correlation import HGNNCorrelationEngine


@dataclass
class InferenceOutput:
    predictions: pd.Series
    raw_result: Any


class V3CorrelationEngine:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        pure_unsupervised: bool = True,
        clustering_method: str = "hdbscan",
        **kwargs: Any,
    ) -> None:
        if not pure_unsupervised:
            raise ValueError("MITRE-CORE V3 requires pure_unsupervised=True at inference.")
        if clustering_method == "prototype":
            raise ValueError("Prototype inference is not permitted in MITRE-CORE V3.")
        self._engine = HGNNCorrelationEngine(
            model_path=model_path,
            device=device,
            pure_unsupervised=True,
            clustering_method=clustering_method,
            use_uf_refinement=False,
            **kwargs,
        )

    def infer(self, alerts: pd.DataFrame) -> InferenceOutput:
        result = self._engine.correlate(alerts)
        if "pred_cluster" not in result.columns:
            raise ValueError("Expected `pred_cluster` in inference output.")
        return InferenceOutput(predictions=result["pred_cluster"], raw_result=result)

    def extract_embeddings(self, alerts: pd.DataFrame):
        return self._engine.extract_embeddings(alerts)
