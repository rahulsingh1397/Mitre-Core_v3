import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv(r'datasets/nsl_kdd/mitre_format.csv')
split_path = Path(r'benchmark/splits/nsl_kdd_10000_seed142.npy')
split_indices = np.load(split_path)
df_eval = df.iloc[split_indices].copy()

print('=== EVAL SPLIT LABEL AUDIT ===')
print(f'Total rows in eval split: {len(df_eval)}')
print()

for col in ['tactic', 'alert_type', 'stage']:
    print(f'--- {col} ---')
    print(f'Unique values: {df_eval[col].nunique()}')
    print(df_eval[col].value_counts().head(10))
    print()

print('--- campaign_id ---')
print(f'Unique values: {df_eval["campaign_id"].nunique()}')
print(df_eval['campaign_id'].value_counts().head(10))
print()

print('=== CROSS-TABULATIONS ===')
print('tactic vs alert_type:')
print(pd.crosstab(df_eval['tactic'], df_eval['alert_type'], margins=True))
print()

print('tactic vs stage (top values):')
ct = pd.crosstab(df_eval['tactic'], df_eval['stage'])
print(ct)
print()

print('campaign_id vs tactic (top pairs):')
ct2 = pd.crosstab(df_eval['campaign_id'], df_eval['tactic'])
print(ct2)
print()

campaign_tactics = df_eval.groupby('campaign_id')['tactic'].nunique()
multi_tactic_campaigns = campaign_tactics[campaign_tactics > 1]
print(f'Campaign IDs spanning multiple tactics: {len(multi_tactic_campaigns)}')
if len(multi_tactic_campaigns) > 0:
    print(multi_tactic_campaigns.head(10))
