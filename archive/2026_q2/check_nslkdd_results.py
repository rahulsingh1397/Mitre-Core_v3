import pandas as pd

df = pd.read_csv('experiments/results/network_v9_nslkdd.csv')
print('NSL-KDD Zero-Shot Results (Network v9):')
print(df[['gate_value', 'ari', 'nmi', 'n_clusters', 'avg_confidence']].to_string(index=False))
print()
best_ari = df.loc[df['ari'].idxmax()]
print(f'Best NSL-KDD zero-shot ARI: {best_ari["ari"]:.4f} at gate={best_ari["gate_value"]}')
print(f'NMI: {best_ari["nmi"]:.4f}, Clusters: {best_ari["n_clusters"]}')
