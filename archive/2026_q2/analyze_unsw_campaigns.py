import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score
from collections import Counter

# Analyze UNSW-NB15 campaign distribution
df = pd.read_csv('datasets/unsw_nb15/mitre_format.csv')
print('Dataset Statistics:')
print(f'Total alerts: {len(df)}')
print(f'Unique campaigns: {df["campaign_id"].nunique()}')

# Campaign size distribution
campaign_sizes = df.groupby('campaign_id').size().sort_values(ascending=False)
print(f'\nCampaign sizes:')
print(campaign_sizes.head(10))
print(f'\nCampaign size statistics:')
print(f'Mean: {campaign_sizes.mean():.1f}')
print(f'Median: {campaign_sizes.median():.1f}')
print(f'Min: {campaign_sizes.min()}')
print(f'Max: {campaign_sizes.max()}')

# Check if campaigns have distinct features
print(f'\nCampaign feature analysis:')
for campaign_id in campaign_sizes.head(5).index:
    campaign_df = df[df['campaign_id'] == campaign_id]
    print(f'\nCampaign {campaign_id} ({len(campaign_df)} alerts):')
    print(f'  Tactics: {sorted(campaign_df["tactic"].unique())}')
    print(f'  Services: {sorted(campaign_df["service"].unique())}')
    print(f'  Unique src_ips: {campaign_df["src_ip"].nunique()}')
    print(f'  Unique dst_ips: {campaign_df["dst_ip"].nunique()}')

# Calculate baseline ARI if we cluster by campaign_id directly
y_true = df['campaign_id'].values
# Random clustering baseline
y_random = np.random.randint(0, df['campaign_id'].nunique(), len(df))
baseline_ari = adjusted_rand_score(y_true, y_random)
print(f'\nBaseline random ARI: {baseline_ari:.4f}')

# Perfect clustering ARI
perfect_ari = adjusted_rand_score(y_true, y_true)
print(f'Perfect clustering ARI: {perfect_ari:.4f}')

# Check feature overlap between campaigns
print(f'\nFeature overlap analysis:')
all_tactics = set(df['tactic'].unique())
all_services = set(df['service'].unique())

for campaign_id in campaign_sizes.head(3).index:
    campaign_df = df[df['campaign_id'] == campaign_id]
    tactics = set(campaign_df['tactic'].unique())
    services = set(campaign_df['service'].unique())
    
    other_campaigns = df[df['campaign_id'] != campaign_id]
    other_tactics = set(other_campaigns['tactic'].unique())
    other_services = set(other_campaigns['service'].unique())
    
    tactic_overlap = len(tactics & other_tactics) / len(tactics | other_tactics)
    service_overlap = len(services & other_services) / len(services | other_services)
    
    print(f'Campaign {campaign_id}:')
    print(f'  Tactic overlap with others: {tactic_overlap:.2f}')
    print(f'  Service overlap with others: {service_overlap:.2f}')
