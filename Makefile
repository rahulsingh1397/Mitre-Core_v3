PYTHON ?= python

benchmark:
	$(PYTHON) -m benchmark.run_benchmark --output benchmark/results/benchmark_results.csv

benchmark-real:
	$(PYTHON) -m benchmark.run_benchmark --datasets benchmark/datasets_real.yaml --output benchmark/results/benchmark_real_results.csv

test:
	pytest tests -q

figures:
	$(PYTHON) -m benchmark.run_benchmark --output benchmark/results/benchmark_results.csv --generate-figures
