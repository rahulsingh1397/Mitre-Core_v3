"""13B: UF Ablation Sweep — UF on vs off across NSL-KDD, UNSW, TON_IoT."""
import subprocess, sys, os
from pathlib import Path

os.chdir(Path(__file__).parent.parent)

datasets = ["NSL-KDD", "UNSW-NB15", "TON_IoT"]
checkpoint = "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
out_dir = Path("experiments/results/ablation_studies")
out_dir.mkdir(parents=True, exist_ok=True)

for ds in datasets:
    for uf_val, uf_label in [(False, "uf_off"), (True, "uf_on")]:
        out_path = out_dir / f"uf_ablation_{ds}_{uf_label}.csv"
        cmd = [
            sys.executable, "experiments/run_gate_tuning.py",
            "--checkpoint", checkpoint,
            "--output", str(out_path),
            "--datasets", ds,
            "--seed", "42",
            "--use_uf_refinement", str(uf_val).lower(),
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr[-500:]}")
        print(f"Output: {out_path}")

print("\n13B complete.")
