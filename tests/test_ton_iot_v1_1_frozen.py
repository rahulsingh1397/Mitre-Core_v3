"""
Frozen baseline integrity test for TON_IoT v1.1.

Verifies:
1. Dataset and checkpoint on-disk hashes match the frozen manifest.
2. Frozen split files exist and their hashes match.
3. Dev/eval split disjointness.
4. (On-demand) V3 lane re-run on the eval split reproduces frozen metrics.

Skip marker: @pytest.mark.slow — excluded from normal CI.
Run with: pytest -m slow tests/test_ton_iot_v1_1_frozen.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
FROZEN_DIR = REPO_ROOT / "benchmark" / "results" / "frozen" / "ton_iot" / "v1.1"
MANIFEST_PATH = FROZEN_DIR / "manifest.json"

# Frozen metric targets (eval split, mean over seeds 42/43/44, alert_type track)
# NOTE: V3 is now competitive on TON_IoT — gap to K-Means(raw) closed from -0.199 to -0.018 ARI.
# These targets are for V3 reproducibility.
FROZEN_V3_ARI_MEAN = 0.6042
FROZEN_V3_AMI_MEAN = 0.8194
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
            "ton_iot_10000_seed42.npy", "ton_iot_10000_seed42.json",
            "ton_iot_10000_seed142.npy", "ton_iot_10000_seed142.json",
        ]
        missing = [f for f in required if not (FROZEN_DIR / f).exists()]
        assert not missing, f"Missing frozen artifacts: {missing}"


class TestHashIntegrity:
    def _ton_iot_entry(self, manifest: dict) -> dict:
        for ds in manifest.get("datasets", []):
            if "ton" in ds.get("name", "").lower():
                return ds
        pytest.skip("TON-IoT not found in manifest datasets")

    def test_checkpoint_hash_matches(self, manifest):
        ds = self._ton_iot_entry(manifest)
        checkpoint_path = Path(ds["checkpoint_path"])
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not on disk: {checkpoint_path}")
        assert sha256_file(checkpoint_path) == ds["checkpoint_sha256"], \
            "Checkpoint SHA256 mismatch"

    def test_dataset_hash_matches(self, manifest):
        ds = self._ton_iot_entry(manifest)
        dataset_path = Path(ds["source_path"])
        if not dataset_path.exists():
            pytest.skip(f"Dataset not on disk: {dataset_path}")
        assert sha256_file(dataset_path) == ds["source_sha256"], \
            "Dataset SHA256 mismatch"

    def test_eval_split_hash_matches(self, manifest):
        ds = self._ton_iot_entry(manifest)
        split_path = Path(ds["split_indices_path"])
        assert split_path.exists(), f"Eval split missing: {split_path}"
        assert sha256_file(split_path) == ds["split_indices_sha256"], \
            "Eval split SHA256 mismatch"


class TestDevEvalDisjoint:
    def test_dev_eval_splits_are_disjoint(self):
        dev_path = REPO_ROOT / "benchmark" / "splits" / "ton_iot_10000_seed42.npy"
        eval_path = REPO_ROOT / "benchmark" / "splits" / "ton_iot_10000_seed142.npy"
        if not dev_path.exists() or not eval_path.exists():
            pytest.skip("Split files not on disk")
        dev_idx = set(np.load(dev_path).tolist())
        eval_idx = set(np.load(eval_path).tolist())
        overlap = dev_idx & eval_idx
        assert not overlap, f"Dev/eval splits overlap at {len(overlap)} indices"


@pytest.mark.slow
class TestFrozenMetricsReproducible:
    """
    Note: V3 is now competitive on TON_IoT (gap to K-Means raw closed to -0.018 ARI).
    These tests verify V3 reproducibility.
    """

    def _run_v3_on_ton_iot(self) -> "pd.DataFrame":
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
            (results["dataset"] == "TON-IoT-exp2.5") &
            (results["label_track"] == "label_col")
        ].copy()

    def test_v3_ari_reproduces(self):
        v3 = self._run_v3_on_ton_iot()
        v3_ari = v3["ari"].mean()
        assert abs(v3_ari - FROZEN_V3_ARI_MEAN) <= TOLERANCE, (
            f"V3 ARI {v3_ari:.4f} deviates from frozen {FROZEN_V3_ARI_MEAN:.4f}"
        )

    def test_v3_ami_reproduces(self):
        v3 = self._run_v3_on_ton_iot()
        v3_ami = v3["ami"].mean()
        assert abs(v3_ami - FROZEN_V3_AMI_MEAN) <= TOLERANCE, (
            f"V3 AMI {v3_ami:.4f} deviates from frozen {FROZEN_V3_AMI_MEAN:.4f}"
        )
