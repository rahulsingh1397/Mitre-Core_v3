from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
import hdbscan
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

from mitre_core.evaluation.manifest import build_run_manifest, sha256_file
from mitre_core.evaluation.unsupervised_metrics import compute_unsupervised_metrics
from mitre_core.inference.correlation_engine import V3CorrelationEngine


@dataclass
class BenchmarkRecord:
    dataset: str
    method: str
    seed: int
    ami: float
    ari: float
    binary_ari: float
    purity: float
    silhouette_cosine: float
    attack_f1: float
    cluster_attribution_f1: float
    latency_seconds_per_10k: float
    peak_gpu_gb: float


@dataclass
class LoadedDataset:
    frame: pd.DataFrame
    features: np.ndarray
    labels: np.ndarray
    benign_label: str | int
    metadata: dict[str, Any]
    label_tracks: dict[str, np.ndarray] | None = None


def reject_label_dependent_inference(infer_fn: Callable) -> None:
    code = getattr(infer_fn, "__code__", None)
    if code is None:
        return
    forbidden = {"y", "y_true", "labels", "ground_truth"}
    if any(name in forbidden for name in code.co_varnames):
        raise ValueError("Inference functions that require labels are forbidden in MITRE-CORE V3.")


def _synthetic_dataset(seed: int, size: int = 120) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = np.asarray([[0.0, 0.0], [3.0, 3.0], [-3.0, 3.0]])
    labels = np.repeat(np.arange(len(centers)), size // len(centers))
    features = np.vstack([rng.normal(loc=center, scale=0.45, size=(size // len(centers), 2)) for center in centers])
    df = pd.DataFrame(
        {
            "AlertId": [f"alert_{i}" for i in range(len(features))],
            "feature_0": features[:, 0],
            "feature_1": features[:, 1],
            "campaign_id": labels,
            "tactic": labels,
            "protocol": 6,
            "service": 80,
        }
    )
    return df, features, labels


def _detect_benign_label(labels: np.ndarray) -> str | int:
    for label in pd.unique(labels):
        normalized = str(label).strip().lower()
        if normalized in {"none", "normal", "benign", "benign traffic"}:
            return label
    return 0


def _sample_indices(df: pd.DataFrame, label_col: str, sample_size: int | None, stratified: bool, seed: int) -> np.ndarray:
    if sample_size is None or len(df) <= sample_size:
        return df.index.to_numpy(dtype=int)
    if stratified and label_col in df.columns and df[label_col].nunique(dropna=False) > 1:
        labels = df[label_col].fillna("UNKNOWN").astype(str)
        try:
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=seed)
            sample_indices, _ = next(splitter.split(df, labels))
            return df.index.to_numpy(dtype=int)[sample_indices]
        except ValueError:
            pass
    return df.sample(n=sample_size, random_state=seed).index.to_numpy(dtype=int)


def _split_directory(dataset: dict[str, Any]) -> Path:
    return Path(dataset.get("split_dir", "benchmark/splits"))


def _split_group_name(dataset: dict[str, Any]) -> str:
    raw_name = str(dataset.get("split_group", dataset["name"]))
    return re.sub(r"[^a-zA-Z0-9]+", "_", raw_name).strip("_").lower() or "dataset"


def _split_path(dataset: dict[str, Any], sample_size: int | None, sample_seed: int) -> Path:
    size_label = "full" if sample_size is None else str(sample_size)
    return _split_directory(dataset) / f"{_split_group_name(dataset)}_{size_label}_seed{sample_seed}.npy"


def _split_metadata_path(split_path: Path) -> Path:
    return split_path.with_suffix(".json")


def _write_split_metadata(split_path: Path, metadata: dict[str, Any]) -> None:
    _split_metadata_path(split_path).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _load_split_metadata(split_path: Path) -> dict[str, Any] | None:
    metadata_path = _split_metadata_path(split_path)
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _materialize_split_indices(
    df: pd.DataFrame,
    *,
    dataset: dict[str, Any],
    dataset_hash: str | None,
    label_col: str,
    sample_size: int | None,
    stratified: bool,
    sample_seed: int,
    excluded_indices: np.ndarray | None,
) -> tuple[np.ndarray, Path]:
    split_path = _split_path(dataset, sample_size, sample_seed)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _load_split_metadata(split_path)
    if split_path.exists():
        indices = np.load(split_path).astype(int)
        if metadata is None:
            metadata = {
                "dataset_sha256": dataset_hash,
                "source_row_count": int(len(df)),
                "sample_size": sample_size,
                "sample_seed": int(sample_seed),
                "label_col": label_col,
                "split_group": _split_group_name(dataset),
            }
            _write_split_metadata(split_path, metadata)
        if metadata.get("dataset_sha256") != dataset_hash:
            raise ValueError(f"Persisted split hash mismatch for {split_path}.")
        if int(metadata.get("source_row_count", -1)) != int(len(df)):
            raise ValueError(f"Persisted split row count mismatch for {split_path}.")
    else:
        excluded_indices = np.unique(excluded_indices.astype(int)) if excluded_indices is not None and excluded_indices.size else np.asarray([], dtype=int)
        remaining_df = df.drop(index=excluded_indices, errors="ignore") if excluded_indices.size else df
        if sample_size is not None and len(remaining_df) < sample_size:
            raise ValueError(f"Not enough rows remaining to create disjoint split for {split_path}.")
        indices = _sample_indices(
            remaining_df,
            label_col=label_col,
            sample_size=sample_size,
            stratified=stratified,
            seed=sample_seed,
        ).astype(int)
        np.save(split_path, indices)
        metadata = {
            "dataset_sha256": dataset_hash,
            "source_row_count": int(len(df)),
            "sample_size": sample_size,
            "sample_seed": int(sample_seed),
            "label_col": label_col,
            "split_group": _split_group_name(dataset),
        }
        _write_split_metadata(split_path, metadata)
    excluded_indices = np.unique(excluded_indices.astype(int)) if excluded_indices is not None and excluded_indices.size else np.asarray([], dtype=int)
    if excluded_indices.size and np.intersect1d(indices, excluded_indices).size:
        raise ValueError(f"Persisted split {split_path} overlaps excluded indices.")
    return indices, split_path


def _resolve_split_indices(
    df: pd.DataFrame,
    *,
    dataset: dict[str, Any],
    dataset_hash: str | None,
    label_col: str,
    sample_size: int | None,
    stratified: bool,
    sample_seed: int,
) -> tuple[np.ndarray, Path | None, list[str]]:
    exclude_sample_seeds = [int(value) for value in dataset.get("exclude_sample_seeds", [])]
    persist_split = "sample_seed" in dataset or bool(exclude_sample_seeds)
    if not persist_split:
        indices = _sample_indices(df, label_col=label_col, sample_size=sample_size, stratified=stratified, seed=sample_seed).astype(int)
        return indices, None, []
    excluded_indices = []
    excluded_paths: list[str] = []
    for excluded_seed in exclude_sample_seeds:
        prior_indices, prior_path = _materialize_split_indices(
            df,
            dataset=dataset,
            dataset_hash=dataset_hash,
            label_col=label_col,
            sample_size=sample_size,
            stratified=stratified,
            sample_seed=excluded_seed,
            excluded_indices=None,
        )
        excluded_indices.append(prior_indices)
        excluded_paths.append(str(prior_path.resolve()))
    excluded_union = np.unique(np.concatenate(excluded_indices)) if excluded_indices else np.asarray([], dtype=int)
    indices, split_path = _materialize_split_indices(
        df,
        dataset=dataset,
        dataset_hash=dataset_hash,
        label_col=label_col,
        sample_size=sample_size,
        stratified=stratified,
        sample_seed=sample_seed,
        excluded_indices=excluded_union,
    )
    return indices, split_path, excluded_paths


def _encode_feature_column(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() >= max(int(len(series) * 0.8), 1):
        return numeric.fillna(0.0).astype(float)
    datetimes = pd.to_datetime(series, errors="coerce", utc=True)
    if datetimes.notna().sum() >= max(int(len(series) * 0.8), 1):
        timestamps = pd.Series(datetimes.view("int64") / 1_000_000_000, index=series.index)
        timestamps[datetimes.isna()] = 0.0
        return timestamps.astype(float)
    codes, _ = pd.factorize(series.fillna("UNKNOWN").astype(str), sort=True)
    return pd.Series(codes, index=series.index, dtype=float)


def _featurize_tabular_dataset(df: pd.DataFrame, label_columns: set[str]) -> np.ndarray:
    excluded = {"AlertId", "EndDate", "correlation_method", "pred_cluster"} | label_columns
    feature_columns = [column for column in df.columns if column not in excluded]
    if not feature_columns:
        raise ValueError("No feature columns available for benchmark dataset.")
    encoded_columns = [_encode_feature_column(df[column]).rename(column) for column in feature_columns]
    features_df = pd.concat(encoded_columns, axis=1)
    features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features_df.to_numpy(dtype=float)


def _load_real_dataset(dataset: dict[str, Any], seed: int) -> LoadedDataset:
    path = Path(dataset["path"])
    if not path.exists():
        raise FileNotFoundError(f"Benchmark dataset not found: {path}")
    if path.suffix == ".parquet":
        source_df = pd.read_parquet(path)
    else:
        source_df = pd.read_csv(path)
    label_col = dataset["label_col"]
    sample_seed = int(dataset.get("sample_seed", seed))
    sample_indices, split_path, excluded_paths = _resolve_split_indices(
        source_df,
        dataset=dataset,
        dataset_hash=sha256_file(path),
        label_col=label_col,
        sample_size=dataset.get("sample_size"),
        stratified=dataset.get("stratified_sample", False),
        sample_seed=sample_seed,
    )
    dataset_hash = sha256_file(path)
    df = source_df.iloc[sample_indices].reset_index(drop=True)
    labels = df[label_col].fillna("UNKNOWN").astype(str).to_numpy()
    label_columns = {label_col}
    if dataset.get("label_col_alt"):
        label_columns.add(dataset["label_col_alt"])
    features = _featurize_tabular_dataset(df, label_columns=label_columns)
    benign_label = _detect_benign_label(labels)
    alt_label_cols = dataset.get("alt_label_cols", [])
    label_tracks: dict[str, np.ndarray] | None = None
    if alt_label_cols:
        label_tracks = {}
        for alt_col in alt_label_cols:
            if alt_col in df.columns:
                label_tracks[alt_col] = df[alt_col].fillna("UNKNOWN").astype(str).to_numpy()
    metadata = {
        "name": dataset["name"],
        "kind": dataset["kind"],
        "source_path": str(path.resolve()),
        "source_sha256": dataset_hash,
        "source_row_count": int(len(source_df)),
        "selected_row_count": int(len(df)),
        "label_col": label_col,
        "sample_size": dataset.get("sample_size"),
        "sample_seed": sample_seed,
        "exclude_sample_seeds": [int(value) for value in dataset.get("exclude_sample_seeds", [])],
        "split_group": _split_group_name(dataset),
        "split_indices_path": str(split_path.resolve()) if split_path is not None else None,
        "split_indices_sha256": sha256_file(split_path) if split_path is not None else None,
        "excluded_split_paths": excluded_paths,
    }
    return LoadedDataset(frame=df, features=features, labels=labels, benign_label=benign_label, metadata=metadata, label_tracks=label_tracks)


def _load_dataset(dataset: dict[str, Any], seed: int) -> LoadedDataset:
    if dataset.get("kind", "synthetic") == "synthetic":
        df, features, labels = _synthetic_dataset(seed)
        metadata = {
            "name": dataset["name"],
            "kind": "synthetic",
            "source_path": None,
            "source_sha256": None,
            "source_row_count": int(len(df)),
            "selected_row_count": int(len(df)),
            "label_col": "campaign_id",
            "sample_size": None,
            "sample_seed": seed,
            "exclude_sample_seeds": [],
            "split_group": dataset["name"],
            "split_indices_path": None,
            "split_indices_sha256": None,
            "excluded_split_paths": [],
        }
        return LoadedDataset(frame=df, features=features, labels=labels, benign_label=0, metadata=metadata)
    if dataset.get("kind") == "mitre_format":
        return _load_real_dataset(dataset, seed)
    raise ValueError(f"Unsupported dataset kind: {dataset.get('kind')}")


def _run_baseline(method: str, features: np.ndarray, seed: int, dataset: dict[str, Any], n_clusters: int, embeddings: np.ndarray | None = None) -> np.ndarray:
    data = embeddings if embeddings is not None else features
    if method == "kmeans_raw":
        return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(features)
    if method == "dbscan_raw":
        return DBSCAN(eps=dataset.get("dbscan_eps", 0.7), min_samples=5).fit_predict(features)
    if method == "hdbscan_raw":
        return hdbscan.HDBSCAN(min_cluster_size=dataset.get("hdbscan_min_cluster_size", 5)).fit_predict(features)
    if method == "kmeans_emb":
        return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(data)
    if method == "spectral_emb":
        from sklearn.cluster import SpectralClustering
        return SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity="nearest_neighbors").fit_predict(data)
    if method == "hdbscan_emb":
        return hdbscan.HDBSCAN(min_cluster_size=dataset.get("hdbscan_min_cluster_size", 5), metric="euclidean").fit_predict(data)
    if method == "spectral_raw":
        from sklearn.cluster import SpectralClustering
        data_std = StandardScaler().fit_transform(features)
        return SpectralClustering(n_clusters=n_clusters, random_state=seed, affinity="nearest_neighbors").fit_predict(data_std)
    if method == "pca_kmeans":
        data_std = StandardScaler().fit_transform(features)
        pca_components = min(dataset.get("pca_components", 16), data_std.shape[1])
        data_pca = PCA(n_components=pca_components, random_state=seed).fit_transform(data_std)
        return KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(data_pca)
    if method == "pca_hdbscan":
        data_std = StandardScaler().fit_transform(features)
        pca_components = min(dataset.get("pca_components", 16), data_std.shape[1])
        data_pca = PCA(n_components=pca_components, random_state=seed).fit_transform(data_std)
        return hdbscan.HDBSCAN(min_cluster_size=dataset.get("hdbscan_min_cluster_size", 5)).fit_predict(data_pca)
    raise ValueError(f"Unknown baseline method: {method}")


def run_benchmark(
    methods_path: Path,
    datasets_path: Path,
    output_path: Path,
    generate_figures: bool = False,
    command: list[str] | None = None,
) -> pd.DataFrame:
    run_start = perf_counter()
    methods = yaml.safe_load(methods_path.read_text(encoding="utf-8"))["methods"]
    datasets = yaml.safe_load(datasets_path.read_text(encoding="utf-8"))["datasets"]
    rows: list[dict] = []
    manifest_datasets: list[dict[str, Any]] = []
    for dataset in datasets:
        if not dataset.get("enabled", True):
            continue
        dataset_name = dataset["name"]
        seeds = dataset.get("seeds", [42, 43, 44])
        dataset_manifest: dict[str, Any] | None = None
        for seed in seeds:
            loaded = _load_dataset(dataset, seed)
            df = loaded.frame
            features = loaded.features
            labels = loaded.labels
            benign_label = loaded.benign_label
            label_tracks = loaded.label_tracks or {}
            all_tracks = {"label_col": labels, **label_tracks}
            n_clusters = int(dataset.get("n_clusters", len(pd.unique(labels))))
            if dataset_manifest is None:
                checkpoint_path = Path(dataset["checkpoint"]) if dataset.get("checkpoint") else None
                dataset_manifest = {
                    **loaded.metadata,
                    "benchmark_seeds": [int(value) for value in seeds],
                    "n_clusters": n_clusters,
                    "checkpoint_path": str(checkpoint_path.resolve()) if checkpoint_path and checkpoint_path.exists() else (str(checkpoint_path) if checkpoint_path else None),
                    "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else None,
                    "clustering_method": dataset.get("clustering_method", dataset.get("engine_kwargs", {}).get("clustering_method", "hdbscan")),
                    "engine_kwargs": dict(dataset.get("engine_kwargs", {})),
                }
            # Lazily extract HGNN embeddings once per seed if any method needs them
            v3_engine: V3CorrelationEngine | None = None
            v3_embeddings: np.ndarray | None = None
            for method in methods:
                if not method.get("enabled", True):
                    continue
                method_name = method["name"]
                start = perf_counter()
                embeddings = features
                if method["type"] == "baseline":
                    if method["entry"].endswith("_emb"):
                        if v3_embeddings is None:
                            engine_kwargs = dict(dataset.get("engine_kwargs", {}))
                            engine_kwargs.setdefault("seed", seed)
                            clustering_method = dataset.get(
                                "clustering_method",
                                engine_kwargs.pop("clustering_method", "hdbscan"),
                            )
                            v3_engine = V3CorrelationEngine(
                                model_path=dataset.get("checkpoint"),
                                pure_unsupervised=True,
                                clustering_method=clustering_method,
                                **engine_kwargs,
                            )
                            v3_embeddings = v3_engine.extract_embeddings(df)
                        embeddings = v3_embeddings
                        pred = _run_baseline(method["entry"], features, seed, dataset, n_clusters, embeddings=embeddings)
                    else:
                        pred = _run_baseline(method["entry"], features, seed, dataset, n_clusters)
                elif method["type"] == "mitre_core_v3":
                    engine_kwargs = dict(dataset.get("engine_kwargs", {}))
                    engine_kwargs.setdefault("seed", seed)
                    clustering_method = dataset.get(
                        "clustering_method",
                        engine_kwargs.pop("clustering_method", "hdbscan"),
                    )
                    if v3_engine is None:
                        v3_engine = V3CorrelationEngine(
                            model_path=dataset.get("checkpoint"),
                            pure_unsupervised=True,
                            clustering_method=clustering_method,
                            **engine_kwargs,
                        )
                    reject_label_dependent_inference(v3_engine.infer)
                    output = v3_engine.infer(df)
                    pred = output.predictions.to_numpy()
                    if v3_embeddings is None:
                        v3_embeddings = v3_engine.extract_embeddings(df)
                    embeddings = v3_embeddings
                else:
                    raise ValueError(f"Unsupported method type: {method['type']}")
                elapsed = perf_counter() - start
                for track_name, track_labels in all_tracks.items():
                    track_benign = _detect_benign_label(track_labels)
                    metrics = compute_unsupervised_metrics(
                        track_labels,
                        pred,
                        embeddings=embeddings,
                        benign_label=track_benign,
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "method": method_name,
                            "seed": seed,
                            "label_track": track_name,
                            **metrics,
                            "latency_seconds_per_10k": elapsed * (10000.0 / len(df)),
                            "peak_gpu_gb": 0.0,
                        }
                    )
        if dataset_manifest is not None:
            manifest_datasets.append(dataset_manifest)
    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    manifest_methods = [
        {
            "name": method["name"],
            "type": method["type"],
            "entry": method.get("entry"),
        }
        for method in methods
        if method.get("enabled", True)
    ]
    results.attrs["run_manifest"] = build_run_manifest(
        output_path=output_path,
        summary_path=output_path.with_name(output_path.stem + "_summary.csv"),
        methods_path=methods_path,
        datasets_path=datasets_path,
        command=command or [],
        datasets=manifest_datasets,
        methods=manifest_methods,
        wall_time_seconds=perf_counter() - run_start,
        peak_gpu_gb=float(results["peak_gpu_gb"].max()) if not results.empty else 0.0,
    )
    if generate_figures:
        figures_dir = output_path.parent.parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
    return results
