import pandas as pd

df = pd.read_csv('datasets/nsl_kdd/mitre_format.csv')
print(f'NSL-KDD dataset size: {len(df)} alerts')
print(f'Unique campaigns: {df["campaign_id"].nunique()}')
print(f'Campaign sizes:')
print(df.groupby('campaign_id').size().sort_values(ascending=False).head(10))
