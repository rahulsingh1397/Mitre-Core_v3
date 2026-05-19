"""Test spectral clustering vs HDBSCAN on UNSW-NB15 and SQTK_SIEM."""
import subprocess, sys

CHECKPOINT = "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
TESTS = [
    ("UNSW-NB15",         "hdbscan", "experiments/results/unsw_hdbscan_baseline.csv"),
    ("UNSW-NB15",         "spectral", "experiments/results/unsw_spectral.csv"),
    ("SQTK_SIEM_kcluster","hdbscan",  "experiments/results/sqtk_hdbscan_baseline.csv"),
    ("SQTK_SIEM_kcluster","spectral", "experiments/results/sqtk_spectral.csv"),
    ("SQTK_SIEM_kcluster","bgmm",     "experiments/results/sqtk_bgmm.csv"),
]

for dataset, method, out in TESTS:
    cmd = [
        sys.executable, "experiments/run_gate_tuning.py",
        "--checkpoint", CHECKPOINT,
        "--datasets", dataset,
        "--clustering_method", method,
        "--output", out,
    ]
    print(f"Running {dataset} + {method} → {out}")
    subprocess.run(cmd, check=True)

print("Spectral sweep complete.")
