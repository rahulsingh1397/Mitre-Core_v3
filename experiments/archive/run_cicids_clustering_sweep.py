"""Compare HDBSCAN, Spectral, and BGMM on CICIDS2017."""
import subprocess, sys

CHECKPOINT = "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
TESTS = [
    ("hdbscan", "experiments/results/cicids_hdbscan.csv"),
    ("bgmm",    "experiments/results/cicids_bgmm.csv"),
    ("spectral","experiments/results/cicids_spectral.csv"),
]

for method, out in TESTS:
    cmd = [
        sys.executable, "experiments/run_gate_tuning.py",
        "--checkpoint", CHECKPOINT,
        "--datasets", "CICIDS2017",
        "--clustering_method", method,
        "--output", out,
    ]
    print(f"Running CICIDS2017 + {method} → {out}")
    subprocess.run(cmd, check=True)
