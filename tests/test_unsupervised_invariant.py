from __future__ import annotations

import pandas as pd
import pytest

from mitre_core.evaluation.benchmark import reject_label_dependent_inference
from mitre_core.inference.correlation_engine import V3CorrelationEngine


def test_v3_engine_rejects_non_unsupervised_mode() -> None:
    with pytest.raises(ValueError):
        V3CorrelationEngine(pure_unsupervised=False)


def test_v3_engine_rejects_prototype_mode() -> None:
    with pytest.raises(ValueError):
        V3CorrelationEngine(clustering_method="prototype")


def test_inference_function_has_no_label_argument() -> None:
    engine = V3CorrelationEngine(pure_unsupervised=True)
    reject_label_dependent_inference(engine.infer)


def test_infer_returns_prediction_column_shape_without_labels() -> None:
    df = pd.DataFrame(
        {
            "AlertId": ["a1", "a2", "a3"],
            "feature_0": [0.0, 1.0, 2.0],
            "feature_1": [1.0, 2.0, 3.0],
            "tactic": [0, 0, 1],
            "protocol": [6, 6, 6],
            "service": [80, 80, 80],
        }
    )
    assert list(df.columns).count("campaign_id") == 0
