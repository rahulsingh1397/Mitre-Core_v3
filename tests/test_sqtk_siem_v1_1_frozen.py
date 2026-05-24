"""
Frozen baseline integrity test for SQTK_SIEM v1.1.

Verifies:
1. Dataset and checkpoint on-disk hashes match the frozen manifest.
2. Frozen split files exist and their hashes match.
3. Split covers the full 5,100-row corpus (no disjoint dev/eval pair).
4. (On-demand) V3 lane re-run reproduces frozen metrics.

Skip marker: @pytest.mark.slow — excluded from normal CI.
Run with: pytest -m slow tests/test_sqtk_siem_v1_1_frozen.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
FROZEN_DIR = REPO_ROOT / "benchmark" / "results" / "frozen" / "sqtk_siem" / "v1.1"
MANIFEST_PATH = FROZEN_DIR / "manifest.json"

# Frozen metric targets (full corpus, mean over seeds 42/43/44, alert_type track)
# NOTE: V3 WINS on SQTK_SIEM v1.1 — ARI 0.461 vs best baseline PCA+HDBSCAN 0.382.
FROZEN_V3_ARI_MEAN = 0.4608
FROZEN_V3_AMI_MEAN = 0.6423
TOLERANCE = 0.01


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.exists(), f"Frozen manifest missing: {MANIFEST_PATH}"
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class TestFrozenArtifactsExist:
    def test_frozen_dir_exists(self):
        assert FROZEN_DIR.exists(), f"Frozen directory missing: {FROZEN_DIR}"

    def test_required_files_present(self):
        required = [
            "results.csv", "summary.csv", "manifest.json",
            "engine_config.yaml", "checkpoint_sha256.txt", "dataset_sha256.txt",
            "environment.txt",
            "sqtk_siem_5100_seed42.npy", "sqtk_siem_5100_seed42.json",
        ]
        missing = [f for f in required if not (FROZEN_DIR / f).exists()]
        assert not missing, f"Missing frozen artifacts: {missing}"


class TestHashIntegrity:
    def _sqtk_siem_entry(self, manifest: dict) -> dict:
        for ds in manifest.get("datasets", []):
            if "sqtk" in ds.get("name", "").lower():
                return ds
        pytest.skip("SQTK_SIEM not found in manifest datasets")

    def test_checkpoint_hash_matches(self, manifest):
        ds = self._sqtk_siem_entry(manifest)
        checkpoint_path = Path(ds["checkpoint_path"])
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not on disk: {checkpoint_path}")
        assert sha256_file(checkpoint_path) == ds["checkpoint_sha256"], \
            "Checkpoint SHA256 mismatch"

    def test_dataset_hash_matches(self, manifest):
        ds = self._sqtk_siem_entry(manifest)
        dataset_path = Path(ds["source_path"])
        if not dataset_path.exists():
            pytest.skip(f"Dataset not on disk: {dataset_path}")
        assert sha256_file(dataset_path) == ds["source_sha256"], \
            "Dataset SHA256 mismatch"

    def test_split_hash_matches(self, manifest):
        ds = self._sqtk_siem_entry(manifest)
        split_path = Path(ds["split_indices_path"])
        assert split_path.exists(), f"Split missing: {split_path}"
        assert sha256_file(split_path) == ds["split_indices_sha256"], \
            "Split SHA256 mismatch"


class TestSplitCoversFullCorpus:
    def test_split_covers_full_corpus(self):
        split_path = REPO_ROOT / "benchmark" / "splits" / "sqtk_siem_5100_seed42.npy"
        if not split_path.exists():
            pytest.skip("Split file not on disk")
        indices = np.load(split_path)
        assert len(indices) == 5100, f"Expected 5100 indices, got {len(indices)}"
        assert len(set(indices)) == 5100, f"Expected 5100 unique indices, got {len(set(indices))}"


@pytest.mark.slow
class TestFrozenMetricsReproducible:
    """
    V3 WINS on SQTK_SIEM v1.1 — ARI 0.461 vs best baseline PCA+HDBSCAN 0.382.
    These tests verify V3 reproducibility.
    """

    def _run_v3_on_sqtk_siem(self) -> "pd.DataFrame":
        import tempfile
        import pandas as pd
        from mitre_core.evaluation.benchmark import run_benchmark

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            tmp_output = Path(f.name)

        results = run_benchmark(
            methods_path=REPO_ROOT / "benchmark" / "methods.yaml",
            datasets_path=REPO_ROOT / "benchmark" / "datasets_real.yaml",
            output_path=tmp_output,
        )
        return results[
            (results["method"] == "MITRE-CORE V3") &
            (results["dataset"] == "SQTK_SIEM") &
            (results["label_track"] == "label_col")
        ].copy()

    def test_v3_ari_reproduces(self):
        v3 = self._run_v3_on_sqtk_siem()
        v3_ari = v3["ari"].mean()
        assert abs(v3_ari - FROZEN_V3_ARI_MEAN) <= TOLERANCE, (
            f"V3 ARI {v3_ari:.4f} deviates from frozen {FROZEN_V3_ARI_MEAN:.4f}"
        )

    def test_v3_ami_reproduces(self):
        v3 = self._run_v3_on_sqtk_siem()
        v3_ami = v3["ami"].mean()
        assert abs(v3_ami - FROZEN_V3_AMI_MEAN) <= TOLERANCE, (
            f"V3 AMI {v3_ami:.4f} deviates from frozen {FROZEN_V3_AMI_MEAN:.4f}"
        )
