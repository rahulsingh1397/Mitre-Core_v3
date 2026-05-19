"""
scripts/process_full_optc.py
---------------------------
Process the full DARPA OpTC dataset for gate tuning experiments.
Expands beyond the 500 sample records to leverage the complete dataset.
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import logging
from pathlib import Path
from datasets.loaders.darpa_optc_loader import DARPAOpTCLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_full_optc():
    """Process the full OpTC dataset beyond sample limitations."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--optc-base", default=None, help="Path to OpTC NCR directory")
    args, _ = parser.parse_known_args()
    
    loader = DARPAOpTCLoader()
    optc_base = Path(args.optc_base) if args.optc_base else Path("datasets/DARPA_OpTC/OpTCNCR")
    
    logger.info("Starting full OpTC dataset processing...")
    
    # Process full dataset with multi-campaign support
    logger.info("Processing full OpTC dataset with multi-campaign support...")
    ecar_bro_dir = optc_base / "ecar-bro"
    bro_dir = optc_base / "bro"
    logger.info("Loading OpTC dataset from eCAR-Bro: %s and Bro logs: %s", 
               ecar_bro_dir, bro_dir)

    # Load data with folder-based campaign extraction for true temporal mixing
    df_combined = loader.load_and_preprocess(
        ecar_bro_dir=ecar_bro_dir,
        bro_dir=bro_dir,
        sample_size=None,  # Process all files
        extract_campaigns=True  # Use folder-based campaign extraction (eCAR endpoint events → RedTeam)
    )
    
    # No date-based labeling - folder-based extraction creates true temporal mixing
    # RedTeam events come from 23Sep19-red/ folder, Benign from Bro logs (both on same dates)
    
    # Save processed dataset
    out_csv = optc_base / "processed_optc_full.csv"
    out_json = optc_base / "processed_optc_full_stats.json"
    
    df_combined.to_csv(out_csv, index=False)
    
    # Generate processing statistics
    campaign_dist = df_combined['CampaignId'].value_counts().to_dict()
    stats = {
        "total_records": len(df_combined),
        "attack_events": int(df_combined['Is_Attack'].sum()),
        "bridge_edge_records": int(df_combined['BroFlowId'].notna().sum()),
        "unique_campaigns": int(df_combined[df_combined['CampaignId'] != 'Unknown']['CampaignId'].nunique()),
        "date_range": f"{df_combined['EndDate'].min()} to {df_combined['EndDate'].max()}",
        "unique_hosts": int(df_combined['SourceHostName'].nunique()),
        "bridge_edge_percentage": float(df_combined['BroFlowId'].notna().mean()),
        "campaign_distribution": campaign_dist
    }
    
    # Save statistics
    import json
    with open(out_json, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✅ Full OpTC processing complete!")
    logger.info(f"📊 Dataset Statistics:")
    logger.info(f"   Total records: {stats['total_records']:,}")
    logger.info(f"   Attack events: {stats['attack_events']:,}")
    logger.info(f"   Bridge edge records: {stats['bridge_edge_records']:,} ({stats['bridge_edge_percentage']:.1%})")
    logger.info(f"   Unique campaigns: {stats['unique_campaigns']}")
    logger.info(f"   Unique hosts: {stats['unique_hosts']}")
    logger.info(f"   Date range: {stats['date_range']}")
    logger.info(f"💾 Saved to: {out_csv}")
    logger.info(f"📋 Stats saved to: {out_json}")
    
    return df_combined

if __name__ == "__main__":
    process_full_optc()
