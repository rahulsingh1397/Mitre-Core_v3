"""13D: TON_IoT sample size sweep at 2K/5K/10K."""
import sys, os
from pathlib import Path

os.chdir(Path(__file__).parent.parent)
sys.path.insert(0, str(Path.cwd()))

from experiments.run_gate_tuning import DATASET_CONFIG, run_sweep

checkpoint = "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
out_dir = Path("experiments/results/ablation_studies")
out_dir.mkdir(parents=True, exist_ok=True)

for sample_size in [2000, 5000, 10000]:
    # Modify TON_IoT config
    original = DATASET_CONFIG["TON_IoT"]["sample_size"]
    DATASET_CONFIG["TON_IoT"]["sample_size"] = sample_size
    
    out_path = out_dir / f"ton_sample_sweep_{sample_size}.csv"
    print(f"Running TON_IoT sample_size={sample_size} -> {out_path}")
    
    run_sweep(
        checkpoint_path=checkpoint,
        output_path=str(out_path),
        datasets=["TON_IoT"],
        seed=42,
    )
    
    # Restore
    DATASET_CONFIG["TON_IoT"]["sample_size"] = original
    print(f"Done: {out_path}")

print("\n13D complete.")
