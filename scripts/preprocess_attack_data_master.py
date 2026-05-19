import argparse
import logging
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# Add project root to path so we can import from training
sys.path.append(str(Path(__file__).resolve().parent.parent))

from training.attack_data_loaders import SysmonXMLLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("preprocess_attack_data")

def process_malware(source_dir: Path, output_dir: Path, max_events: int):
    logger.info("Processing malware datasets...")
    malware_families = {
        "minergate": "minergate",
        "trickbot/infection": "trickbot",
        "xmrig_miner": "xmrig_miner",
        "conti-cobalt": "conti"
    }
    
    all_dfs = []
    
    for rel_path, family_name in malware_families.items():
        family_dir = source_dir / "datasets" / "malware" / rel_path
        if not family_dir.exists():
            # Try without datasets/malware just in case structure is different
            family_dir = source_dir / "malware" / rel_path
            if not family_dir.exists():
                family_dir = source_dir / rel_path
                if not family_dir.exists():
                    logger.warning(f"Could not find directory for {family_name} at {family_dir}")
                    continue
                
        logger.info(f"Loading {family_name} from {family_dir}")
        loader = SysmonXMLLoader(str(family_dir))
        df = loader.load(limit=max_events)
        
        if not df.empty:
            df["campaign_id"] = family_name
            df["MalwareIntelAttackType"] = family_name
            all_dfs.append(df)
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Ensure MITRE-CORE schema columns
        if "AlertId" not in final_df.columns:
            final_df["AlertId"] = [f"malware_{i}" for i in range(len(final_df))]
        if "EndDate" not in final_df.columns and "Timestamp" in final_df.columns:
            final_df["EndDate"] = final_df["Timestamp"]
            
        output_file = output_dir / "malware_sysmon_mitre_format.parquet"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_file)
        logger.info(f"Saved {len(final_df)} malware events to {output_file}")
        logger.info(f"Families: {final_df['campaign_id'].value_counts().to_dict()}")
    else:
        logger.warning("No malware data processed.")

def process_attack_techniques(source_dir: Path, output_dir: Path, max_events: int):
    logger.info("Processing attack techniques...")
    techniques_dir = source_dir / "attack_techniques"
    if not techniques_dir.exists():
        techniques_dir = source_dir / "datasets" / "attack_techniques"
        if not techniques_dir.exists():
            logger.warning(f"Could not find attack_techniques directory at {techniques_dir}")
            return
            
    all_dfs = []
    
    # Walk T-code folders
    for t_dir in tqdm(list(techniques_dir.iterdir())):
        if not t_dir.is_dir() or not t_dir.name.startswith("T"):
            continue
            
        t_code = t_dir.name
        loader = SysmonXMLLoader(str(t_dir))
        df = loader.load(limit=max_events)
        
        if not df.empty:
            df["campaign_id"] = t_code
            df["MalwareIntelAttackType"] = t_code
            all_dfs.append(df)
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        # Ensure MITRE-CORE schema columns
        if "AlertId" not in final_df.columns:
            final_df["AlertId"] = [f"tech_{i}" for i in range(len(final_df))]
        if "EndDate" not in final_df.columns and "Timestamp" in final_df.columns:
            final_df["EndDate"] = final_df["Timestamp"]
            
        output_file = output_dir / "attack_techniques_mitre_format.parquet"
        output_dir.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(output_file)
        logger.info(f"Saved {len(final_df)} technique events to {output_file}")
        logger.info(f"Techniques: {len(final_df['campaign_id'].unique())}")
    else:
        logger.warning("No attack techniques data processed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess large malware datasets for MITRE-CORE v3")
    parser.add_argument("--source", type=str, required=True, help="Source directory (attack_data-master)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--max-events-per-family", type=int, default=10000, help="Max events per malware family")
    parser.add_argument("--max-events-per-technique", type=int, default=500, help="Max events per attack technique")
    
    args = parser.parse_args()
    
    source_dir = Path(args.source)
    output_dir = Path(args.output_dir)
    
    process_malware(source_dir, output_dir, args.max_events_per_family)
    process_attack_techniques(source_dir, output_dir, args.max_events_per_technique)
    
    logger.info("Preprocessing complete.")
