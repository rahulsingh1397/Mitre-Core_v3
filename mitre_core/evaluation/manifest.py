from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=None)
def _sha256_file_cached(resolved_path: str) -> str:
    digest = hashlib.sha256()
    with open(resolved_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = str(Path(path).resolve())
    if not Path(resolved).exists():
        return None
    return _sha256_file_cached(resolved)


def _runtime_context() -> dict[str, Any]:
    packages = {
        "python": platform.python_version(),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "torch": _package_version("torch"),
        "torch-geometric": _package_version("torch-geometric"),
        "hdbscan": _package_version("hdbscan"),
        "scikit-learn": _package_version("scikit-learn"),
        "pyyaml": _package_version("PyYAML"),
    }
    host = {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_executable": sys.executable,
    }
    gpu: dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "devices": [],
        "peak_gpu_gb": 0.0,
    }
    try:
        import torch

        gpu["available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            gpu["device_count"] = int(torch.cuda.device_count())
            gpu["devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            peak_allocated = max(
                [torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())],
                default=0,
            )
            gpu["peak_gpu_gb"] = float(peak_allocated / (1024**3))
    except Exception:
        pass
    return {
        "packages": packages,
        "host": host,
        "gpu": gpu,
    }


def build_run_manifest(
    *,
    output_path: Path,
    summary_path: Path,
    methods_path: Path,
    datasets_path: Path,
    command: list[str],
    datasets: list[dict[str, Any]],
    methods: list[dict[str, Any]],
    wall_time_seconds: float,
    peak_gpu_gb: float,
) -> dict[str, Any]:
    runtime = _runtime_context()
    runtime["gpu"]["peak_gpu_gb"] = max(float(runtime["gpu"].get("peak_gpu_gb", 0.0)), float(peak_gpu_gb))
    return {
        "command": command,
        "command_line": " ".join(command),
        "results_path": str(output_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "methods_config_path": str(methods_path.resolve()),
        "datasets_config_path": str(datasets_path.resolve()),
        "wall_time_seconds": float(wall_time_seconds),
        "peak_gpu_gb": float(peak_gpu_gb),
        "datasets": datasets,
        "methods": methods,
        **runtime,
    }


def write_run_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
