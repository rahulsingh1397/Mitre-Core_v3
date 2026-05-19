from __future__ import annotations

import argparse
from pathlib import Path
import sys

from mitre_core.evaluation.benchmark import run_benchmark
from mitre_core.evaluation.manifest import write_run_manifest
from mitre_core.evaluation.multi_seed import aggregate_multi_seed


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the MITRE-CORE V3 benchmark suite.")
    parser.add_argument("--methods", default="benchmark/methods.yaml")
    parser.add_argument("--datasets", default="benchmark/datasets.yaml")
    parser.add_argument("--output", default="benchmark/results/benchmark_results.csv")
    parser.add_argument("--generate-figures", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    results = run_benchmark(
        methods_path=Path(args.methods),
        datasets_path=Path(args.datasets),
        output_path=output,
        generate_figures=args.generate_figures,
        command=[sys.executable, "-m", "benchmark.run_benchmark", *sys.argv[1:]],
    )
    summary = aggregate_multi_seed(results)
    summary_path = output.with_name(output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    manifest = results.attrs.get("run_manifest")
    if manifest:
        manifest["summary_path"] = str(summary_path.resolve())
        write_run_manifest(manifest, output.with_name(output.stem + "_manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
