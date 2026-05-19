import sys
import argparse
import pandas as pd
from mitre_core.inference import V3CorrelationEngine

def main():
    parser = argparse.ArgumentParser(description="MITRE-CORE V3 unsupervised alert correlation")
    parser.add_argument("input_csv", help="Path to input raw alerts CSV")
    parser.add_argument("output_csv", help="Path to save correlated alerts CSV")
    parser.add_argument(
        "--checkpoint",
        default="hgnn_checkpoints/network_v9_v3/network_it_best.pt",
        help="Path to an unsupervised or semi-unsupervised HGNN checkpoint",
    )
    args = parser.parse_args()

    print(f"Loading alerts from {args.input_csv}...")
    try:
        df = pd.read_csv(args.input_csv)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        sys.exit(1)

    print(f"Initializing MITRE-CORE V3 engine with {args.checkpoint}...")
    engine = V3CorrelationEngine(
        model_path=args.checkpoint,
        hdbscan_auto_tune=True,
        hdbscan_cluster_selection_epsilon=0.1,
        hdbscan_use_umap=True,
    )

    print("Running correlation (Graph Construction -> HGNN -> HDBSCAN)...")
    results = engine.infer(df).raw_result

    print(f"Saving results to {args.output_csv}...")
    results.to_csv(args.output_csv, index=False)
    print(f"Done! Found {results['pred_cluster'].nunique()} attack campaigns.")

if __name__ == "__main__":
    main()
