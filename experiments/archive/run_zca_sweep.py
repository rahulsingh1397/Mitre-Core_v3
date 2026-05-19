"""Sweep Soft-ZCA epsilon on SQTK_SIEM to find optimal whitening strength."""
import subprocess, sys

EPS_VALUES = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
CHECKPOINT = "hgnn_checkpoints/network_v9_v3/network_it_best.pt"

for eps in EPS_VALUES:
    out = f"experiments/results/sqtk_zca_eps{eps}.csv"
    cmd = [
        sys.executable, "experiments/run_gate_tuning.py",
        "--checkpoint", CHECKPOINT,
        "--datasets", "SQTK_SIEM_kcluster",
        "--zca_whitening", "True",
        "--zca_eps", str(eps),
        "--output", out,
    ]
    print(f"Running ZCA eps={eps} → {out}")
    subprocess.run(cmd, check=True)

print("ZCA sweep complete. Compare ami column across output files.")
