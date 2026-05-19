"""E2E verification on cleaned codebase with canonical checkpoints."""
import sys, os, time, logging, pandas as pd, numpy as np, json
from pathlib import Path
sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING)

from hgnn.hgnn_correlation import HGNNCorrelationEngine
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

results = {}
checkpoints = {
    'network_v9_v3': 'hgnn_checkpoints/network_v9_v3/network_it_best.pt',
    'siem_supcon_v4': 'hgnn_checkpoints/siem_supcon_v4/best.pt',
    'unsw_supcon_v7': 'hgnn_checkpoints/unsw_supcon_v7/best.pt',
}

datasets = {
    'UNSW-NB15': ('datasets/unsw_nb15/mitre_format.csv', 'campaign_id', 2000),
    'NSL-KDD': ('datasets/nsl_kdd/mitre_format.csv', 'campaign_id', 2000),
    'TON_IoT': ('datasets/TON_IoT/mitre_format.parquet', 'campaign_id', 2000),
}

for cp_name, cp_path in checkpoints.items():
    if not Path(cp_path).exists():
        print(f'SKIP {cp_name}: not found')
        continue
    for ds_name, (ds_path, label_col, n) in datasets.items():
        if not Path(ds_path).exists():
            print(f'SKIP {ds_name}: not found')
            continue
        try:
            engine = HGNNCorrelationEngine(
                model_path=cp_path, use_geometric_confidence=True,
                pure_unsupervised=True, hdbscan_auto_tune=True,
                hdbscan_cluster_selection_epsilon=0.1,
                use_uf_refinement=False, seed=42,
            )
            if ds_path.endswith('.parquet'):
                df = pd.read_parquet(ds_path)
            else:
                df = pd.read_csv(ds_path)
            df = df.sample(min(n, len(df)), random_state=42)
            t0 = time.time()
            result = engine.correlate(df)
            elapsed = time.time() - t0
            ari = adjusted_rand_score(df[label_col], result['pred_cluster'])
            ami = adjusted_mutual_info_score(df[label_col], result['pred_cluster'])
            n_clusters = result['pred_cluster'].nunique()
            key = f'{cp_name}/{ds_name}'
            results[key] = {'ari': round(ari, 4), 'ami': round(ami, 4), 'clusters': n_clusters, 'time_s': round(elapsed, 1)}
            print(f'{key}: ARI={ari:.4f}, AMI={ami:.4f}, clusters={n_clusters}, time={elapsed:.1f}s')
        except Exception as e:
            print(f'ERROR {cp_name}/{ds_name}: {e}')
            import traceback
            traceback.print_exc()

print()
print('=' * 60)
print('SUMMARY')
print('=' * 60)
for k, v in results.items():
    print(f'  {k}: ARI={v["ari"]}, AMI={v["ami"]}, clusters={v["clusters"]}')

out_path = Path('experiments/results/e2e_reports/clean_verification.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'Saved to {out_path}')
