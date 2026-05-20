#!/usr/bin/env python3
"""
Verify OpTC ARI=1.0 legitimacy.
Reads processed_optc_unified.csv in chunks — no full extraction needed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

CSV_PATH = "datasets/DARPA_OpTC/OpTCNCR-20260326T025141Z-1-006/OpTCNCR/processed_optc_full.csv"
CHUNKSIZE = 100_000

def check_ip_exclusivity_and_dates():
    """Check if ARI=1.0 comes from structural IP differences or date leakage."""
    print("=== OpTC ARI=1.0 Verification: IP Exclusivity & Date Analysis ===")
    print(f"Reading: {CSV_PATH}")
    print(f"Chunk size: {CHUNKSIZE:,}")
    
    # Initialize tracking sets
    attack_src, benign_src = set(), set()
    attack_dst, benign_dst = set(), set()
    date_campaign = defaultdict(set)
    total = 0
    
    # Stream through CSV in chunks to avoid memory issues
    campaign_values = set()
    for chunk in pd.read_csv(CSV_PATH, chunksize=CHUNKSIZE,
                              usecols=['CampaignId', 'SourceAddress',
                                       'DestinationAddress', 'EndDate'],
                              low_memory=False):
        total += len(chunk)
        
        # Track all unique campaign values
        campaign_values.update(chunk['CampaignId'].dropna().unique())
        
        # Separate attack and benign records dynamically
        campaign_list = list(campaign_values)
        redteam_campaigns = [c for c in campaign_list if 'RedTeam' in str(c)]
        
        if redteam_campaigns:
            atk = chunk[chunk['CampaignId'].isin(redteam_campaigns)]
        else:
            atk = pd.DataFrame()  # Empty if no redteam campaigns found
        
        ben = chunk[chunk['CampaignId'] == 'Benign']

        # Track IP exclusivity
        if not atk.empty:
            attack_src.update(atk['SourceAddress'].dropna())
            attack_dst.update(atk['DestinationAddress'].dropna())
        
        if not ben.empty:
            benign_src.update(ben['SourceAddress'].dropna())
            benign_dst.update(ben['DestinationAddress'].dropna())

        # Track date-campaign mappings
        chunk['date'] = pd.to_datetime(chunk['EndDate']).dt.date
        for date, cids in chunk.groupby('date')['CampaignId']:
            date_campaign[date].update(cids.unique())

        print(f"  Processed {total:,} rows...", end='\r')

    print(f"\nTotal rows: {total:,}")
    
    # Show all unique campaign values found
    print(f"\n=== CAMPAIGN VALUES FOUND ===")
    redteam_campaigns = [c for c in campaign_values if 'RedTeam' in str(c)]
    print(f"All unique CampaignId values: {sorted(campaign_values)}")
    print(f"RedTeam campaigns found: {redteam_campaigns}")

    # IP exclusivity analysis
    print("\n=== IP EXCLUSIVITY ANALYSIS ===")
    for label, a_ips, b_ips in [
        ('SourceAddress', attack_src, benign_src),
        ('DestinationAddress', attack_dst, benign_dst),
    ]:
        exclusive = a_ips - b_ips
        overlap   = a_ips & b_ips
        print(f"\n{label}:")
        print(f"  Attack-exclusive IPs: {len(exclusive):,}/{len(a_ips):,} "
              f"= {len(exclusive)/max(len(a_ips),1):.1%}")
        print(f"  Overlapping IPs:      {len(overlap):,}/{len(a_ips):,} "
              f"= {len(overlap)/max(len(a_ips),1):.1%}")

    # Date distribution analysis
    print("\n=== DATE DISTRIBUTION ANALYSIS ===")
    attack_dates = set()
    benign_dates = set()
    mixed_dates = set()
    
    print("\nDate → Campaign mapping (sorted):")
    for date in sorted(date_campaign.keys()):
        campaigns = date_campaign[date]
        print(f"  {date}: {sorted(campaigns)}")
        
        if 'RedTeam_Sep23' in campaigns and 'Benign' in campaigns:
            mixed_dates.add(date)
        elif 'RedTeam_Sep23' in campaigns:
            attack_dates.add(date)
        else:
            benign_dates.add(date)
    
    # Summary statistics
    print(f"\n=== SUMMARY STATISTICS ===")
    print(f"Total unique dates: {len(date_campaign)}")
    print(f"Attack-only dates: {len(attack_dates)}")
    print(f"Benign-only dates: {len(benign_dates)}")
    print(f"Mixed dates (both campaigns): {len(mixed_dates)}")
    
    # Calculate IP exclusivity percentages
    src_excl_pct = len(attack_src - benign_src) / max(len(attack_src), 1)
    dst_excl_pct = len(attack_dst - benign_dst) / max(len(attack_dst), 1)
    
    print(f"\nSource IP exclusivity: {src_excl_pct:.1%}")
    print(f"Destination IP exclusivity: {dst_excl_pct:.1%}")
    
    # Verdict based on analysis
    print(f"\n=== VERDICT ===")
    
    if mixed_dates:
        print(f"✓ MIXED DATES FOUND: {len(mixed_dates)} dates contain both campaigns")
        print("  → RedTeam attacks overlap temporally with benign traffic")
        
        if src_excl_pct > 0.5 or dst_excl_pct > 0.5:
            print("✓ HIGH IP EXCLUSIVITY: Attack infrastructure is structurally distinct")
            print("  → LIKELY LEGITIMATE: Model detects real attack patterns")
        else:
            print("⚠ LOW IP EXCLUSIVITY: Attack infrastructure overlaps with benign")
            print("  → MIXED SIGNAL: Some pattern detection + some temporal leakage")
            print("  → RECOMMEND: Run Experiment B (temporal split)")
    else:
        print("⚠ NO MIXED DATES: RedTeam operates on completely separate dates")
        print("  → TEMPORAL LEAKAGE: Model likely learned date=Sep23 → attack")
        print("  → RECOMMEND: Run Experiment B (temporal split) to confirm")
    
    # Additional insights
    print(f"\n=== ADDITIONAL INSIGHTS ===")
    if attack_dates:
        print(f"Attack date range: {min(attack_dates)} to {max(attack_dates)}")
    if benign_dates:
        print(f"Benign date range: {min(benign_dates)} to {max(benign_dates)}")
    
    # Check if Sep 23 is exclusive (as expected from name)
    sep23_in_mixed = '2019-09-23' in mixed_dates
    sep23_attack_only = '2019-09-23' in attack_dates
    
    if sep23_attack_only and not sep23_in_mixed:
        print("✓ Sep 23 is attack-only (consistent with RedTeam_Sep23 naming)")
    elif sep23_in_mixed:
        print("⚠ Sep 23 has mixed traffic (unexpected given naming)")
    else:
        print("? Sep 23 not found in data")
    
    return {
        'src_exclusivity': src_excl_pct,
        'dst_exclusivity': dst_excl_pct,
        'mixed_dates': len(mixed_dates),
        'total_dates': len(date_campaign),
        'verdict': 'legitimate' if (mixed_dates and (src_excl_pct > 0.5 or dst_excl_pct > 0.5)) else 'temporal_leakage'
    }

if __name__ == "__main__":
    results = check_ip_exclusivity_and_dates()
