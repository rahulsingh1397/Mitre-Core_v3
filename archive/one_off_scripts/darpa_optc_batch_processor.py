import pandas as pd
import numpy as np
import json
import gzip
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime
import re
import os

logger = logging.getLogger(__name__)

class DARPAOpTCBatchProcessor:
    """
    Efficient batch processor for DARPA OpTC dataset.
    Processes data in manageable chunks to handle the large dataset size.
    """
    
    def __init__(self, root_path: Optional[Union[str, Path]] = None, batch_size: int = 1000):
        """
        Initialize the OpTC batch processor.
        
        Args:
            root_path: Base path to OpTC dataset directory
            batch_size: Number of records to process at once
        """
        self.root_path = Path(root_path) if root_path else None
        self.ecar_bro_path = None
        self.bro_path = None
        self.batch_size = batch_size
        self.processed_records = 0
        
    def set_data_paths(self, ecar_bro_path: Union[str, Path], bro_path: Union[str, Path]):
        """Set paths to OpTC data directories."""
        self.ecar_bro_path = Path(ecar_bro_path)
        self.bro_path = Path(bro_path)
        
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the OpTC dataset without processing all data."""
        stats = {
            'ecar_files': 0,
            'bro_files': 0,
            'estimated_records': 0,
            'date_range': None,
            'hosts': set(),
            'file_sizes': []
        }
        
        if self.ecar_bro_path and self.ecar_bro_path.exists():
            for subset in ['benign', 'evaluation']:
                subset_path = self.ecar_bro_path / subset
                if subset_path.exists():
                    for date_dir in subset_path.iterdir():
                        if date_dir.is_dir():
                            for host_dir in date_dir.iterdir():
                                if host_dir.is_dir():
                                    json_files = list(host_dir.glob("*.json.gz"))
                                    stats['ecar_files'] += len(json_files)
                                    stats['hosts'].add(host_dir.name)
                                    
                                    # Sample a few files to estimate record count
                                    for json_file in json_files[:3]:
                                        try:
                                            with gzip.open(json_file, 'rt') as f:
                                                lines = f.readlines()
                                                stats['estimated_records'] += len(lines)
                                                stats['file_sizes'].append(len(lines))
                                        except Exception as e:
                                            logger.warning(f"Could not sample {json_file}: {e}")
        
        if self.bro_path and self.bro_path.exists():
            for date_dir in self.bro_path.iterdir():
                if date_dir.is_dir():
                    log_files = list(date_dir.glob("**/*.log"))
                    stats['bro_files'] += len(log_files)
        
        stats['hosts'] = len(stats['hosts'])
        return stats
    
    def process_ecar_file_batch(self, json_files: List[Path]) -> pd.DataFrame:
        """Process a batch of eCAR JSON files."""
        all_events = []
        
        for json_file in json_files:
            try:
                with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                    for line in f:
                        try:
                            event = json.loads(line.strip())
                            processed_event = self._process_ecar_event(event)
                            if processed_event:
                                all_events.append(processed_event)
                                
                                if len(all_events) >= self.batch_size:
                                    yield pd.DataFrame(all_events)
                                    all_events = []
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse JSON line: {e}")
                            continue
            except Exception as e:
                logger.error(f"Error processing eCAR file {json_file}: {e}")
                continue
        
        if all_events:
            yield pd.DataFrame(all_events)
    
    def _process_ecar_event(self, event: Dict) -> Optional[Dict]:
        """Process individual eCAR event."""
        try:
            # Basic event extraction
            processed = {
                'timestamp': event.get('timestamp', event.get('time', datetime.now().isoformat())),
                'event_type': event.get('type', 'unknown'),
                'source_ip': None,
                'destination_ip': None,
                'source_host': None,
                'destination_host': None,
                'process_name': None,
                'process_id': None,
                'user': None,
                'command_line': None,
                'file_path': None,
                'network_protocol': None,
                'source_port': None,
                'destination_port': None,
                'bro_flow_id': None,
                'is_attack': 0,
                'attack_type': 'Normal',
                'tactic': 'None'
            }
            
            # Extract subject information
            if 'subject' in event:
                subject = event['subject']
                processed['process_id'] = subject.get('pid')
                processed['user'] = subject.get('user', subject.get('username'))
                processed['source_host'] = subject.get('host', subject.get('hostname'))
            
            # Extract object information
            if 'object' in event:
                obj = event['object']
                obj_type = obj.get('type', '').lower()
                
                if obj_type == 'fileobject':
                    processed['file_path'] = obj.get('path')
                    processed['destination_host'] = obj.get('host')
                elif obj_type == 'netflowobject':
                    processed['destination_ip'] = obj.get('remoteAddress', obj.get('dst_ip'))
                    processed['destination_port'] = obj.get('remotePort', obj.get('dst_port'))
                    processed['network_protocol'] = obj.get('protocol')
                    processed['bro_flow_id'] = obj.get('bro_uid')
                elif obj_type == 'processobject':
                    processed['process_name'] = obj.get('name')
                    processed['command_line'] = obj.get('commandLine')
                    processed['destination_host'] = obj.get('host')
            
            # Extract network flow correlations
            if 'bro_flow_start' in event:
                bro_flow = event['bro_flow_start']
                processed['source_ip'] = bro_flow.get('src_ip')
                processed['source_port'] = bro_flow.get('src_port')
                processed['destination_ip'] = bro_flow.get('dst_ip')
                processed['destination_port'] = bro_flow.get('dst_port')
                processed['network_protocol'] = bro_flow.get('protocol')
                processed['bro_flow_id'] = bro_flow.get('bro_uid')
            
            # Map to MITRE ATT&CK tactics
            processed['tactic'] = self._map_event_to_tactic(processed['event_type'], event)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing eCAR event: {e}")
            return None
    
    def _map_event_to_tactic(self, event_type: str, event_data: Dict) -> str:
        """Map eCAR event types to MITRE ATT&CK tactics."""
        event_type = event_type.lower()
        
        if 'process' in event_type or 'exec' in event_type:
            return 'Execution'
        elif 'file' in event_type:
            if 'create' in event_type or 'write' in event_type:
                return 'Persistence'
            elif 'read' in event_type:
                return 'Collection'
            elif 'delete' in event_type:
                return 'Defense Evasion'
        elif 'network' in event_type or 'flow' in event_type:
            return 'Command and Control'
        elif 'registry' in event_type or 'config' in event_type:
            return 'Persistence'
        elif 'authentication' in event_type or 'logon' in event_type:
            return 'Initial Access'
            
        return 'Execution'
    
    def process_bro_logs_batch(self, bro_dir: Path) -> pd.DataFrame:
        """Process Bro logs in batch."""
        all_flows = []
        log_types = ['conn', 'dns', 'http', 'ssl', 'files']
        
        for log_type in log_types:
            log_files = list(bro_dir.glob(f"**/{log_type}.*.log"))
            logger.info(f"Processing {len(log_files)} {log_type} log files")
            
            for log_file in log_files:
                try:
                    flows = self._parse_bro_log_file(log_file, log_type)
                    all_flows.extend(flows)
                    
                    if len(all_flows) >= self.batch_size:
                        yield pd.DataFrame(all_flows)
                        all_flows = []
                        
                except Exception as e:
                    logger.warning(f"Failed to parse {log_file}: {e}")
                    continue
        
        if all_flows:
            yield pd.DataFrame(all_flows)
    
    def _parse_bro_log_file(self, log_file: Path, log_type: str) -> List[Dict]:
        """Parse individual Bro log file."""
        flows = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('#'):
                        start_idx = i
                        break
                
                for line in lines[start_idx:]:
                    if line.strip() and not line.startswith('#'):
                        flow = self._parse_bro_line(line.strip(), log_type)
                        if flow:
                            flows.append(flow)
        except Exception as e:
            logger.error(f"Error parsing Bro log {log_file}: {e}")
            
        return flows
    
    def _parse_bro_line(self, line: str, log_type: str) -> Optional[Dict]:
        """Parse individual Bro log line."""
        try:
            fields = line.split('\t')
            
            if log_type == 'conn':
                return {
                    'timestamp': fields[0] if len(fields) > 0 else None,
                    'bro_uid': fields[1] if len(fields) > 1 else None,
                    'source_ip': fields[2] if len(fields) > 2 else None,
                    'source_port': fields[3] if len(fields) > 3 else None,
                    'destination_ip': fields[4] if len(fields) > 4 else None,
                    'destination_port': fields[5] if len(fields) > 5 else None,
                    'protocol': fields[6] if len(fields) > 6 else None,
                    'service': fields[7] if len(fields) > 7 else None,
                    'duration': fields[8] if len(fields) > 8 else None,
                    'source_bytes': fields[9] if len(fields) > 9 else None,
                    'destination_bytes': fields[10] if len(fields) > 10 else None,
                    'conn_state': fields[11] if len(fields) > 11 else None,
                    'log_type': 'conn'
                }
            elif log_type == 'dns':
                return {
                    'timestamp': fields[0] if len(fields) > 0 else None,
                    'bro_uid': fields[1] if len(fields) > 1 else None,
                    'source_ip': fields[2] if len(fields) > 2 else None,
                    'destination_ip': fields[4] if len(fields) > 4 else None,
                    'query': fields[9] if len(fields) > 9 else None,
                    'qtype': fields[11] if len(fields) > 11 else None,
                    'rcode': fields[15] if len(fields) > 15 else None,
                    'log_type': 'dns'
                }
            elif log_type == 'http':
                return {
                    'timestamp': fields[0] if len(fields) > 0 else None,
                    'bro_uid': fields[1] if len(fields) > 1 else None,
                    'source_ip': fields[2] if len(fields) > 2 else None,
                    'destination_ip': fields[4] if len(fields) > 4 else None,
                    'method': fields[7] if len(fields) > 7 else None,
                    'host': fields[8] if len(fields) > 8 else None,
                    'uri': fields[9] if len(fields) > 9 else None,
                    'status_code': fields[11] if len(fields) > 11 else None,
                    'log_type': 'http'
                }
                
        except Exception as e:
            logger.debug(f"Failed to parse Bro line: {e}")
            return None
            
        return None
    
    def convert_to_mitre_schema(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Convert processed data to MITRE-CORE schema."""
        mitre_df = pd.DataFrame()
        
        if data_type == 'ecar':
            # eCAR events
            mitre_df['SourceAddress'] = df['source_ip'].fillna('0.0.0.0')
            mitre_df['DestinationAddress'] = df['destination_ip'].fillna('0.0.0.0')
            mitre_df['DeviceAddress'] = '172.16.254.1'
            mitre_df['SourceHostName'] = df['source_host'].fillna(df['source_ip'].apply(lambda x: f"host-{str(x).replace('.', '-')}"))
            mitre_df['DestinationHostName'] = df['destination_host'].fillna(df['destination_ip'].apply(lambda x: f"target-{str(x).replace('.', '-')}"))
            mitre_df['DeviceHostName'] = 'optc-sensor-alpha'
            mitre_df['SourceUserName'] = df['user'].fillna('unknown_user')
            mitre_df['EndDate'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp.now())
            mitre_df['MalwareIntelAttackType'] = df['attack_type'].fillna('Normal')
            mitre_df['AttackSeverity'] = 10
            mitre_df['Is_Attack'] = df['is_attack']
            mitre_df['AlertId'] = 'OpTC_' + df.index.astype(str) + '_' + mitre_df['EndDate'].astype(str)
            mitre_df['ProcessName'] = df['process_name']
            mitre_df['ProcessId'] = df['process_id']
            mitre_df['CommandLine'] = df['command_line']
            mitre_df['FilePath'] = df['file_path']
            mitre_df['NetworkProtocol'] = df['network_protocol']
            mitre_df['SourcePort'] = df['source_port']
            mitre_df['DestinationPort'] = df['destination_port']
            mitre_df['Tactic'] = df['tactic']
            mitre_df['BroFlowId'] = df['bro_flow_id']
            
        elif data_type == 'bro':
            # Bro network flows
            mitre_df['SourceAddress'] = df['source_ip'].fillna('0.0.0.0')
            mitre_df['DestinationAddress'] = df['destination_ip'].fillna('0.0.0.0')
            mitre_df['DeviceAddress'] = '172.16.254.2'
            mitre_df['SourceHostName'] = df['source_ip'].apply(lambda x: f"host-{str(x).replace('.', '-')}")
            mitre_df['DestinationHostName'] = df['destination_ip'].apply(lambda x: f"target-{str(x).replace('.', '-')}")
            mitre_df['DeviceHostName'] = 'bro-sensor-alpha'
            mitre_df['SourceUserName'] = 'network_service'
            mitre_df['EndDate'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp.now())
            mitre_df['MalwareIntelAttackType'] = 'Network_Flow'
            mitre_df['AttackSeverity'] = 1
            mitre_df['Is_Attack'] = 0
            mitre_df['Tactic'] = 'None'
            mitre_df['AlertId'] = 'Bro_' + df.index.astype(str) + '_' + mitre_df['EndDate'].astype(str)
            mitre_df['NetworkProtocol'] = df['protocol']
            mitre_df['SourcePort'] = df['source_port']
            mitre_df['DestinationPort'] = df['destination_port']
            mitre_df['BroFlowId'] = df['bro_uid']
        
        return mitre_df
    
    def process_full_dataset(self, ecar_bro_dir: Optional[Union[str, Path]] = None, 
                           bro_dir: Optional[Union[str, Path]] = None,
                           max_records: Optional[int] = None,
                           save_intermediate: bool = True) -> pd.DataFrame:
        """
        Process full OpTC dataset efficiently.
        
        Args:
            ecar_bro_dir: Directory containing eCAR-Bro JSON files
            bro_dir: Directory containing Bro logs
            max_records: Maximum number of records to process
            save_intermediate: Save intermediate results
            
        Returns:
            DataFrame in MITRE-CORE schema
        """
        ecar_bro_dir = Path(ecar_bro_dir) if ecar_bro_dir else self.ecar_bro_path
        bro_dir = Path(bro_dir) if bro_dir else self.bro_path
        
        if not ecar_bro_dir or not bro_dir:
            raise ValueError("Both ecar_bro_dir and bro_dir must be specified")
        
        logger.info("Starting efficient OpTC dataset processing...")
        
        # Get dataset statistics first
        stats = self.get_dataset_statistics()
        logger.info(f"Dataset statistics: {stats}")
        
        all_data = []
        record_count = 0
        
        # Process eCAR events in batches
        logger.info("Processing eCAR events...")
        if ecar_bro_dir.exists():
            for subset in ['benign', 'evaluation']:
                subset_path = ecar_bro_dir / subset
                if subset_path.exists():
                    logger.info(f"Processing {subset} subset...")
                    
                    # Collect all JSON files
                    all_json_files = []
                    for date_dir in subset_path.iterdir():
                        if date_dir.is_dir():
                            for host_dir in date_dir.iterdir():
                                if host_dir.is_dir():
                                    json_files = list(host_dir.glob("*.json.gz"))
                                    all_json_files.extend(json_files)
                    
                    logger.info(f"Found {len(all_json_files)} JSON files in {subset}")
                    
                    # Process in batches
                    for batch_df in self.process_ecar_file_batch(all_json_files):
                        mitre_df = self.convert_to_mitre_schema(batch_df, 'ecar')
                        all_data.append(mitre_df)
                        record_count += len(mitre_df)
                        
                        logger.info(f"Processed {record_count} eCAR records")
                        
                        if max_records and record_count >= max_records:
                            break
                    
                    if max_records and record_count >= max_records:
                        break
        
        # Process Bro flows in batches
        logger.info("Processing Bro flows...")
        if bro_dir.exists() and (not max_records or record_count < max_records):
            for batch_df in self.process_bro_logs_batch(bro_dir):
                mitre_df = self.convert_to_mitre_schema(batch_df, 'bro')
                all_data.append(mitre_df)
                record_count += len(mitre_df)
                
                logger.info(f"Processed {record_count} total records")
                
                if max_records and record_count >= max_records:
                    break
        
        if not all_data:
            logger.warning("No data loaded from OpTC dataset")
            return pd.DataFrame()
        
        # Combine all data
        logger.info("Combining all processed data...")
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates and sort by timestamp
        combined_df = combined_df.drop_duplicates(subset=['AlertId'])
        combined_df = combined_df.sort_values('EndDate').reset_index(drop=True)
        
        logger.info(f"Final dataset: {len(combined_df)} records from {record_count} processed")
        
        return combined_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example usage
    processor = DARPAOpTCBatchProcessor(batch_size=5000)
    
    # Set data paths
    optc_base = Path("E:/Private/MITRE-CORE 2/MITRE-CORE_V2/datasets/DARPA_OpTC/OpTCNCR-20260326T025141Z-1-006/OpTCNCR")
    ecar_bro_path = optc_base / "ecar-bro"
    bro_path = optc_base / "bro"
    
    # Get dataset statistics first
    logger.info("Getting dataset statistics...")
    stats = processor.get_dataset_statistics()
    print(f"Dataset Statistics: {stats}")
    
    # Process a sample of the dataset
    logger.info("Processing OpTC dataset sample...")
    try:
        df = processor.process_full_dataset(ecar_bro_path, bro_path, max_records=5000, save_intermediate=True)
        
        print(f"\nProcessed {len(df)} OpTC records")
        print(f"Date range: {df['EndDate'].min()} to {df['EndDate'].max()}")
        print(f"Attack events: {df['Is_Attack'].sum()}")
        print(f"Unique hosts: {df['SourceHostName'].nunique()}")
        
        print("\nSample data:")
        print(df[['EndDate', 'SourceAddress', 'DestinationAddress', 'MalwareIntelAttackType', 'Tactic', 'ProcessName']].head())
        
        print(f"\nAttack types: {df['MalwareIntelAttackType'].value_counts()}")
        print(f"\nTactics: {df['Tactic'].value_counts()}")
        
        # Save processed data
        output_path = "E:/Private/MITRE-CORE 2/MITRE-CORE_V2/datasets/DARPA_OpTC/processed_optc_sample.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved processed data to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to process OpTC data: {e}")
