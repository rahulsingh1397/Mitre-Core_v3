"""Regression guards for production ARI targets.

These tests pin the canonical per-dataset ARI floors that MITRE-CORE must
preserve. Run via `pytest tests/test_regression_aris.py -v`.

Each test reads the latest final production sweep CSV and asserts that the
best ARI observed for the dataset meets the documented floor. If the floor
is missed, either (a) the CSV is stale and must be regenerated, or (b) a
regression was introduced and must be fixed before merge.
"""
from pathlib import Path

import pandas as pd
import pytest

SWEEP_CSV = Path("experiments/results/zeroshot_baseline_final.csv")

# Per-dataset ARI floors (see docs/architecture/PRODUCTION_ARCHITECTURE_V3.md).
FLOORS = {
    "UNSW-NB15": 0.50,
    "NSL-KDD": 0.72,
    "TON_IoT": 0.07,  # Fragmentation issue, low floor
    "OpTC": 0.04,  # Domain shift challenges
    "CICIDS2017": 0.25,
    "SQTK_SIEM_kcluster": 0.10,  # Embedding collapse
}


def _load_sweep() -> pd.DataFrame:
    if not SWEEP_CSV.exists():
        pytest.skip(f"{SWEEP_CSV} not generated yet; run experiments/run_gate_tuning.py")
    return pd.read_csv(SWEEP_CSV)


@pytest.mark.parametrize("dataset,floor", list(FLOORS.items()))
def test_ari_floor(dataset: str, floor: float) -> None:
    df = _load_sweep()
    sub = df[df["dataset"] == dataset]
    if sub.empty:
        pytest.skip(f"{dataset} not present in {SWEEP_CSV}")
    best = float(sub["ari"].max())
    assert best >= floor, (
        f"{dataset} ARI regressed: observed best={best:.4f}, floor={floor:.2f}. "
        "Check recent changes to run_gate_tuning.py, hgnn_correlation.py, or "
        "the checkpoint for this dataset."
    )
