"""
Frozen baseline integrity test for NSL-KDD v1.0.

Verifies:
1. Dataset and checkpoint on-disk hashes match the frozen manifest.
2. Frozen split files exist and their hashes match.
3. (On-demand) V3 lane re-run on the eval split reproduces frozen metrics.

Skip marker: @pytest.mark.slow — the re-run test is excluded from normal CI
and is only run via `pytest -m slow` or `make test-frozen`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
FROZEN_DIR = REPO_ROOT / "benchmark" / "results" / "frozen" / "nsl_kdd" / "v1.0"
MANIFEST_PATH = FROZEN_DIR / "manifest.json"

# Frozen metric targets (eval split, mean over seeds 42/43/44)
FROZEN_V3_ARI_MEAN = 0.6025
FROZEN_V3_AMI_MEAN = 0.6852
TOLERANCE = 0.01  # allow ±1% for floating-point nondeterminism


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
            "nsl_kdd_10000_seed42.npy", "nsl_kdd_10000_seed42.json",
            "nsl_kdd_10000_seed142.npy", "nsl_kdd_10000_seed142.json",
        ]
        missing = [f for f in required if not (FROZEN_DIR / f).exists()]
        assert not missing, f"Missing frozen artifacts: {missing}"


class TestHashIntegrity:
    def test_checkpoint_hash_matches(self, manifest):
        ds = manifest["datasets"][0]
        checkpoint_path = Path(ds["checkpoint_path"])
        if not checkpoint_path.exists():
            pytest.skip(f"Checkpoint not on disk: {checkpoint_path}")
        assert sha256_file(checkpoint_path) == ds["checkpoint_sha256"], \
            "Checkpoint SHA256 mismatch — checkpoint may have been replaced"

    def test_dataset_hash_matches(self, manifest):
        ds = manifest["datasets"][0]
        dataset_path = Path(ds["source_path"])
        if not dataset_path.exists():
            pytest.skip(f"Dataset not on disk: {dataset_path}")
        assert sha256_file(dataset_path) == ds["source_sha256"], \
            "Dataset SHA256 mismatch — source CSV may have been modified"

    def test_eval_split_hash_matches(self, manifest):
        ds = manifest["datasets"][0]
        split_path = Path(ds["split_indices_path"])
        assert split_path.exists(), f"Eval split missing: {split_path}"
        assert sha256_file(split_path) == ds["split_indices_sha256"], \
            "Eval split SHA256 mismatch — split indices may have been regenerated"


class TestDevEvalDisjoint:
    def test_dev_eval_splits_are_disjoint(self):
        dev_path = REPO_ROOT / "benchmark" / "splits" / "nsl_kdd_10000_seed42.npy"
        eval_path = REPO_ROOT / "benchmark" / "splits" / "nsl_kdd_10000_seed142.npy"
        if not dev_path.exists() or not eval_path.exists():
            pytest.skip("Split files not on disk")
        dev_idx = set(np.load(dev_path).tolist())
        eval_idx = set(np.load(eval_path).tolist())
        overlap = dev_idx & eval_idx
        assert not overlap, f"Dev/eval splits overlap at {len(overlap)} indices"


@pytest.mark.slow
class TestFrozenMetricsReproducible:
    """
    Runs the full V3 eval lane and asserts frozen metrics reproduce.
    Excluded from normal CI — run with: pytest -m slow tests/test_nsl_kdd_v1_frozen.py
    """

    def _run_v3_on_nsl_kdd(self) -> "pd.DataFrame":
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
        return results[results["method"] == "MITRE-CORE V3"].copy()

    def test_v3_ari_reproduces(self):
        v3 = self._run_v3_on_nsl_kdd()
        v3_ari = v3["ari"].mean()
        assert abs(v3_ari - FROZEN_V3_ARI_MEAN) <= TOLERANCE, (
            f"V3 ARI {v3_ari:.4f} deviates from frozen {FROZEN_V3_ARI_MEAN:.4f} "
            f"by more than tolerance {TOLERANCE}"
        )

    def test_v3_ami_reproduces(self):
        v3 = self._run_v3_on_nsl_kdd()
        v3_ami = v3["ami"].mean()
        assert abs(v3_ami - FROZEN_V3_AMI_MEAN) <= TOLERANCE, (
            f"V3 AMI {v3_ami:.4f} deviates from frozen {FROZEN_V3_AMI_MEAN:.4f} "
            f"by more than tolerance {TOLERANCE}"
        )
