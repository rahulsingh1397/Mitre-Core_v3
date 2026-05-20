"""
Asserts that every frozen dataset version directory contains a manifest.json.
Prevents silent publishing of results without provenance.
"""
from __future__ import annotations

from pathlib import Path

import pytest

FROZEN_ROOT = Path(__file__).parent.parent / "benchmark" / "results" / "frozen"


def _frozen_version_dirs() -> list[Path]:
    """Return all vX.Y subdirectories under benchmark/results/frozen/."""
    if not FROZEN_ROOT.exists():
        return []
    return [
        p for p in FROZEN_ROOT.rglob("*")
        if p.is_dir() and p.name.startswith("v") and p.parent.parent == FROZEN_ROOT
    ]


@pytest.mark.parametrize("version_dir", _frozen_version_dirs())
def test_frozen_version_has_manifest(version_dir: Path) -> None:
    manifest = version_dir / "manifest.json"
    assert manifest.exists(), (
        f"Frozen directory {version_dir} is missing manifest.json.\n"
        "Every published frozen result must include a manifest with full provenance.\n"
        "See docs/plans/MASTER_PLAN_v1.0.md Part V.3 for the contract."
    )


@pytest.mark.parametrize("version_dir", _frozen_version_dirs())
def test_frozen_version_has_summary(version_dir: Path) -> None:
    summary = version_dir / "summary.csv"
    assert summary.exists(), (
        f"Frozen directory {version_dir} is missing summary.csv."
    )


@pytest.mark.parametrize("version_dir", _frozen_version_dirs())
def test_frozen_version_has_engine_config(version_dir: Path) -> None:
    cfg = version_dir / "engine_config.yaml"
    assert cfg.exists(), (
        f"Frozen directory {version_dir} is missing engine_config.yaml.\n"
        "Engine config must be locked alongside results so reproduction is possible."
    )
