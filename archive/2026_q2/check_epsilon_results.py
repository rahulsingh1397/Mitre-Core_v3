import pandas as pd

df = pd.read_csv('experiments/results/network_v9_evaluation_epsilon_sweep.csv')
print('Epsilon Sweep Results for UNSW-NB15:')
print(df[['epsilon', 'ari', 'n_clusters', 'avg_confidence']].to_string(index=False))
print()
best_idx = df['ari'].idxmax()
best_row = df.loc[best_idx]
print(f'Best epsilon: {best_row["epsilon"]} (ARI={best_row["ari"]:.4f})')
