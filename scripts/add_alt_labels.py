import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_temporal_sessions(df: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    """Group alerts into sessions via inactivity gap per src_ip."""
    if 'timestamp' not in df.columns or 'src_ip' not in df.columns:
        logger.warning("Missing timestamp or src_ip. Cannot create temporal sessions.")
        df['alt_campaign_id_v2'] = df.get('campaign_id', 0)
        return df

    df = df.sort_values(['src_ip', 'timestamp'])
    
    # Calculate time difference between consecutive alerts from the same IP
    df['time_diff'] = df.groupby('src_ip')['timestamp'].diff()
    
    # A new session starts if the gap is larger than gap_minutes
    # or if it's the first alert for that IP (time_diff is NaT)
    is_new_session = (df['time_diff'] > pd.Timedelta(minutes=gap_minutes)) | df['time_diff'].isna()
    
    # Assign session IDs (cumulative sum of new session flags)
    df['alt_campaign_id_v2'] = is_new_session.cumsum()
    
    df = df.drop(columns=['time_diff']).sort_index()
    logger.info(f"Created {df['alt_campaign_id_v2'].nunique()} temporal sessions.")
    return df

def main():
    parser = argparse.ArgumentParser(description="Add independent alt_ari labels to dataset")
    parser.add_argument("--input", type=str, required=True, help="Input parquet/csv file")
    parser.add_argument("--output", type=str, required=True, help="Output parquet/csv file")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
        
    if input_path.suffix == '.parquet':
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    
    df = add_temporal_sessions(df)
    
    if args.output.endswith('.parquet'):
        df.to_parquet(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)
        
    logger.info(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
