import pandas as pd
import numpy as np
import json
import gzip
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DARPAOpTCLoader:
    """
    Unified loader and preprocessor for DARPA Operationally Transparent Cyber (OpTC) dataset.
    
    This is the single, authoritative loader for OpTC data that consolidates all functionality
    from previous redundant processor files. Handles eCAR endpoint events and Bro network logs,
    mapping to MITRE-CORE schema with optional campaign labeling for multi-stage APT detection.
    """
    
    def __init__(self, root_path: Optional[Union[str, Path]] = None):
        """Initialize the unified OpTC loader."""
        self.root_path = Path(root_path) if root_path else None
        self.ecar_bro_path = None
        self.bro_path = None
        self.campaign_counter = 1
        
    def set_data_paths(self, ecar_bro_path: Union[str, Path], bro_path: Union[str, Path]):
        """Set paths to OpTC data directories."""
        self.ecar_bro_path = Path(ecar_bro_path)
        self.bro_path = Path(bro_path)
        
    def load_and_preprocess(self, ecar_bro_dir: Optional[Union[str, Path]] = None, 
                           bro_dir: Optional[Union[str, Path]] = None,
                           sample_size: Optional[int] = None,
                           extract_campaigns: bool = True) -> pd.DataFrame:
        """Load and preprocess OpTC dataset into MITRE-CORE compatible format."""
        ecar_bro_dir = Path(ecar_bro_dir) if ecar_bro_dir else self.ecar_bro_path
        bro_dir = Path(bro_dir) if bro_dir else self.bro_path
        
        if not ecar_bro_dir or not bro_dir:
            raise ValueError("Both ecar_bro_dir and bro_dir must be specified")
            
        logger.info(f"Loading OpTC dataset from eCAR-Bro: {ecar_bro_dir} and Bro logs: {bro_dir}")
        
        # Process eCAR-Bro endpoint events
        ecar_data = self._process_ecar_events(ecar_bro_dir, sample_size)
        
        # Process Bro network flows
        bro_data = self._process_bro_flows(bro_dir, sample_size)
        
        # Combine and convert to MITRE-CORE schema
        all_data = []
        
        if not ecar_data.empty:
            mitre_ecar = self._convert_ecar_to_mitre_schema(ecar_data)
            all_data.append(mitre_ecar)
            
        if not bro_data.empty:
            mitre_bro = self._convert_bro_to_mitre_schema(bro_data)
            all_data.append(mitre_bro)
            
        if not all_data:
            logger.warning("No data loaded from OpTC dataset")
            return pd.DataFrame()
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['AlertId'])
        combined_df = combined_df.sort_values('EndDate').reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined_df)} records from OpTC dataset")
        
        # Extract campaign labels if requested
        if extract_campaigns:
            combined_df = self._extract_campaign_labels(combined_df)
            
        return combined_df
    
    def _process_ecar_events(self, ecar_bro_path: Path, max_records: Optional[int] = None) -> pd.DataFrame:
        """Process eCAR-Bro JSON files."""
        all_events = []
        
        if not ecar_bro_path.exists():
            logger.warning(f"eCAR-Bro path does not exist: {ecar_bro_path}")
            return pd.DataFrame()
        
        # Process evaluation data first (contains attacks)
        eval_path = ecar_bro_path / "evaluation"
        if eval_path.exists():
            logger.info(f"Processing evaluation subset: {eval_path}")
            events = self._process_ecar_subset(eval_path, max_records)
            all_events.append(events)
            
        # Process benign data if room available
        benign_path = ecar_bro_path / "benign"
        if benign_path.exists():
            logger.info(f"Processing benign subset: {benign_path}")
            remaining = max_records - sum(len(d) for d in all_events) if max_records else None
            events = self._process_ecar_subset(benign_path, remaining)
            all_events.append(events)
            
        if not all_events:
            return pd.DataFrame()
            
        return pd.concat(all_events, ignore_index=True) if len(all_events) > 1 else all_events[0]
    
    def _process_ecar_subset(self, subset_path: Path, max_records: Optional[int] = None) -> pd.DataFrame:
        """Process a subset (evaluation/benign) of eCAR data."""
        all_events = []
        
        date_dirs = [d for d in subset_path.iterdir() if d.is_dir()]
        
        for date_dir in date_dirs:
            host_dirs = [h for h in date_dir.iterdir() if h.is_dir()][:3]  # Limit hosts
            
            for host_dir in host_dirs:
                json_files = list(host_dir.glob("*.json.gz"))[:2]  # Limit files
                
                for json_file in json_files:
                    try:
                        events = self._parse_ecar_json_file(json_file)
                        if not events.empty:
                            all_events.append(events)
                            
                    except Exception as e:
                        logger.error(f"Error processing {json_file}: {e}")
                        continue
                        
        if not all_events:
            return pd.DataFrame()
            
        return pd.concat(all_events, ignore_index=True) if len(all_events) > 1 else all_events[0]
    
    def _parse_ecar_json_file(self, json_file: Path) -> pd.DataFrame:
        """Parse individual eCAR JSON file."""
        events = []
        
        try:
            with gzip.open(json_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        line = line.strip()
                        if line:
                            event = json.loads(line)
                            processed = self._extract_ecar_event(event)
                            if processed:
                                events.append(processed)
                    except json.JSONDecodeError:
                        continue
                    except Exception:
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing eCAR file {json_file}: {e}")
            
        return pd.DataFrame(events)
    
    def _extract_ecar_event(self, event: Dict) -> Optional[Dict]:
        """Extract relevant fields from eCAR event for MITRE-CORE schema."""
        try:
            # Validate that event is a dictionary
            if not isinstance(event, dict):
                logger.debug(f"Skipping non-dictionary event: {type(event)}")
                return None
                
            extracted = {
                'timestamp': event.get('timestamp', event.get('time', datetime.now().isoformat())),
                'event_type': event.get('type', 'unknown'),
                'source_ip': None, 'destination_ip': None, 'source_host': None, 'destination_host': None,
                'process_name': None, 'process_id': None, 'user': None, 'command_line': None,
                'file_path': None, 'network_protocol': None, 'source_port': None, 'destination_port': None,
                'bro_flow_id': None, 'is_attack': 0, 'attack_type': 'Normal', 'tactic': 'None'
            }
            
            # Extract subject information
            if 'subject' in event and isinstance(event['subject'], dict):
                subject = event['subject']
                extracted['process_id'] = subject.get('pid')
                extracted['user'] = subject.get('user', subject.get('username'))
                extracted['source_host'] = subject.get('host', subject.get('hostname'))
                
            # Extract object information
            if 'object' in event and isinstance(event['object'], dict):
                obj = event['object']
                obj_type = obj.get('type', '').lower()
                
                if obj_type == 'fileobject':
                    extracted['file_path'] = obj.get('path')
                    extracted['destination_host'] = obj.get('host')
                elif obj_type == 'netflowobject':
                    extracted['destination_ip'] = obj.get('remoteAddress', obj.get('dst_ip'))
                    extracted['destination_port'] = obj.get('remotePort', obj.get('dst_port'))
                    extracted['network_protocol'] = obj.get('protocol')
                    extracted['bro_flow_id'] = obj.get('bro_uid')
                elif obj_type == 'processobject':
                    extracted['process_name'] = obj.get('name')
                    extracted['command_line'] = obj.get('commandLine')
                    extracted['destination_host'] = obj.get('host')
            
            # Extract network flow correlations
            if 'bro_flow_start' in event and isinstance(event['bro_flow_start'], dict):
                bro_flow = event['bro_flow_start']
                extracted['source_ip'] = bro_flow.get('src_ip')
                extracted['source_port'] = bro_flow.get('src_port')
                extracted['destination_ip'] = bro_flow.get('dst_ip')
                extracted['destination_port'] = bro_flow.get('dst_port')
                extracted['network_protocol'] = bro_flow.get('protocol')
                extracted['bro_flow_id'] = bro_flow.get('bro_uid')
                
            # Map to MITRE ATT&CK tactics
            extracted['tactic'] = self._map_event_to_tactic(extracted['event_type'])
            
            return extracted
            
        except Exception as e:
            logger.error(f"Error processing eCAR event: {e}")
            return None
    
    def _map_event_to_tactic(self, event_type: str) -> str:
        """Map eCAR event types to MITRE ATT&CK tactics."""
        event_type = event_type.lower()
        
        if 'process' in event_type or 'exec' in event_type:
            return 'Execution'
        elif 'file' in event_type:
            if 'create' in event_type or 'write' in event_type:
                return 'Persistence'
        elif 'network' in event_type or 'flow' in event_type:
            return 'Command and Control'
        elif 'registry' in event_type or 'config' in event_type:
            return 'Persistence'
        elif 'authentication' in event_type or 'logon' in event_type:
            return 'Initial Access'
            
        return 'Execution'
    
    def _process_bro_flows(self, bro_path: Path, max_records: Optional[int] = None) -> pd.DataFrame:
        """
        Process Bro/Zeek network logs.
        
        Args:
            bro_path: Directory containing Bro log files
            max_records: Optional limit on number of records
            
        Returns:
            DataFrame with parsed network flows
        """
        bro_dir = Path(bro_path)
        all_flows = []
        
        # Process different Bro log types
        log_types = ['conn', 'dns', 'http', 'ssl', 'files']
        
        for log_type in log_types:
            log_files = list(bro_dir.glob(f"**/{log_type}.*.log"))
            for log_file in log_files:
                try:
                    flows = self._parse_bro_log_file(log_file, log_type)
                    all_flows.extend(flows)
                except Exception as e:
                    logger.warning(f"Failed to parse {log_file}: {e}")
                    continue
                    
        return pd.DataFrame(all_flows)
    
    def _parse_bro_log_file(self, log_file: Path, log_type: str) -> List[Dict]:
        """Parse individual Bro log file."""
        flows = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                # Skip Bro header lines
                lines = f.readlines()
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('#'):
                        start_idx = i
                        break
                
                # Parse log entries
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
                # Connection log: ts uid id.orig_h id.orig_p id.resp_h id.resp_p proto service duration orig_bytes resp_bytes conn_state local_orig local_resp missed_bytes history orig_pkts orig_ip_bytes resp_pkts resp_ip_bytes tunnel_parents
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
                # DNS log parsing
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
                # HTTP log parsing
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
    
    def load_and_preprocess(self, ecar_bro_dir: Optional[Union[str, Path]] = None, 
                           bro_dir: Optional[Union[str, Path]] = None,
                           sample_size: Optional[int] = None,
                           process_full_dataset: bool = True,
                           extract_campaigns: bool = True) -> pd.DataFrame:
        """
        Load and preprocess OpTC dataset into MITRE-CORE compatible format.
        
        Args:
            ecar_bro_dir: Directory containing eCAR-Bro JSON files
            bro_dir: Directory containing Bro logs
            sample_size: Maximum number of records to process
            process_full_dataset: If True, process all files instead of sample
            extract_campaigns: If True, extract campaign labels for multi-stage APT detection
            
        Returns:
            DataFrame in MITRE-CORE schema with optional campaign labels
        """
        ecar_bro_dir = Path(ecar_bro_dir) if ecar_bro_dir else self.ecar_bro_path
        bro_dir = Path(bro_dir) if bro_dir else self.bro_path
        
        if not ecar_bro_dir or not bro_dir:
            raise ValueError("Both ecar_bro_dir and bro_dir must be specified")
            
        logger.info(f"Loading OpTC dataset from eCAR-Bro: {ecar_bro_dir} and Bro logs: {bro_dir}")
        
        # Load eCAR-Bro endpoint events
        ecar_events = []
        if ecar_bro_dir.exists():
            for subset in ['benign', 'evaluation']:
                subset_path = ecar_bro_dir / subset
                if subset_path.exists():
                    logger.info(f"Processing {subset} subset: {subset_path}")
                    
                    date_dirs = list(subset_path.iterdir())
                    logger.info(f"Found {len(date_dirs)} date directories")
                    
                    for date_dir in date_dirs:
                        if date_dir.is_dir():
                            logger.info(f"Processing date directory: {date_dir.name}")
                            
                            host_dirs = list(date_dir.iterdir())
                            logger.info(f"Found {len(host_dirs)} host directories")
                            
                            for host_dir in host_dirs:
                                if host_dir.is_dir():
                                    json_files = list(host_dir.glob("*.json.gz"))
                                    logger.info(f"Host {host_dir.name}: {len(json_files)} JSON files")
                                    
                                    # Process all files if process_full_dataset=True, otherwise limit
                                    files_to_process = json_files if process_full_dataset else json_files[:5]
                                    
                                    for json_file in files_to_process:
                                        logger.info(f"Processing eCAR file: {json_file}")
                                        df = self._parse_ecar_json_file(json_file)
                                        if not df.empty:
                                            ecar_events.append(df)
                                            logger.info(f"Loaded {len(df)} events from {json_file}")
                                            
                                            if sample_size and sum(len(d) for d in ecar_events) >= sample_size:
                                                logger.info(f"Reached sample size limit: {sample_size}")
                                                break
                                if sample_size and sum(len(d) for d in ecar_events) >= sample_size:
                                    break
                            if sample_size and sum(len(d) for d in ecar_events) >= sample_size:
                                break
                        if sample_size and sum(len(d) for d in ecar_events) >= sample_size:
                            break
                    if sample_size and sum(len(d) for d in ecar_events) >= sample_size:
                        break
                        
        # Load Bro network logs
        bro_flows = []
        if bro_dir.exists():
            date_dirs = list(bro_dir.iterdir())
            logger.info(f"Found {len(date_dirs)} Bro date directories")
            
            for date_dir in date_dirs:
                if date_dir.is_dir():
                    logger.info(f"Processing Bro logs from: {date_dir}")
                    df = self._process_bro_flows(date_dir)
                    if not df.empty:
                        bro_flows.append(df)
                        logger.info(f"Loaded {len(df)} Bro flows from {date_dir}")
                        
                        if sample_size and sum(len(d) for d in bro_flows) >= sample_size // 2:
                            logger.info(f"Reached Bro sample size limit: {sample_size // 2}")
                            break
                            
        # Combine and convert to MITRE-CORE schema
        all_data = []
        
        # Process eCAR events
        for df in ecar_events:
            if not df.empty:
                mitre_df = self._convert_ecar_to_mitre_schema(df)
                all_data.append(mitre_df)
                
        # Process Bro flows
        for df in bro_flows:
            if not df.empty:
                mitre_df = self._convert_bro_to_mitre_schema(df)
                all_data.append(mitre_df)
                
        if not all_data:
            logger.warning("No data loaded from OpTC dataset")
            return pd.DataFrame()
            
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates and sort by timestamp
        combined_df = combined_df.drop_duplicates(subset=['AlertId'])
        combined_df = combined_df.sort_values('EndDate').reset_index(drop=True)
        
        logger.info(f"Loaded {len(combined_df)} records from OpTC dataset")
        
        # Extract campaign labels if requested
        if extract_campaigns:
            combined_df = self._extract_campaign_labels(combined_df)
            
        return combined_df
    
    def _convert_ecar_to_mitre_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert eCAR events to MITRE-CORE schema."""
        mitre_df = pd.DataFrame()
        
        # Map network addresses
        mitre_df['SourceAddress'] = df['source_ip'].fillna('0.0.0.0')
        mitre_df['DestinationAddress'] = df['destination_ip'].fillna('0.0.0.0')
        mitre_df['DeviceAddress'] = '172.16.254.1'  # Simulated sensor IP
        
        # Map hostnames
        mitre_df['SourceHostName'] = df['source_host'].fillna(df['source_ip'].apply(lambda x: f"host-{str(x).replace('.', '-')}"))
        mitre_df['DestinationHostName'] = df['destination_host'].fillna(df['destination_ip'].apply(lambda x: f"target-{str(x).replace('.', '-')}"))
        mitre_df['DeviceHostName'] = 'optc-sensor-alpha'
        
        # Map user information
        mitre_df['SourceUserName'] = df['user'].fillna('unknown_user')
        
        # Map timestamps
        mitre_df['EndDate'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp.now())
        
        # Map attack information
        mitre_df['MalwareIntelAttackType'] = df['attack_type'].fillna('Normal')
        mitre_df['AttackSeverity'] = 10  # Default severity
        mitre_df['Is_Attack'] = df['is_attack']
        
        # Generate unique alert IDs
        mitre_df['AlertId'] = 'OpTC_' + df.index.astype(str) + '_' + mitre_df['EndDate'].astype(str)
        
        # Add additional context
        mitre_df['ProcessName'] = df['process_name']
        mitre_df['ProcessId'] = df['process_id']
        mitre_df['CommandLine'] = df['command_line']
        mitre_df['FilePath'] = df['file_path']
        mitre_df['NetworkProtocol'] = df['network_protocol']
        mitre_df['SourcePort'] = df['source_port']
        mitre_df['DestinationPort'] = df['destination_port']
        mitre_df['Tactic'] = df['tactic']
        mitre_df['BroFlowId'] = df['bro_flow_id']
        
        return mitre_df
    
    def _convert_bro_to_mitre_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert Bro network flows to MITRE-CORE schema."""
        mitre_df = pd.DataFrame()
        
        # Map network addresses
        mitre_df['SourceAddress'] = df['source_ip'].fillna('0.0.0.0')
        mitre_df['DestinationAddress'] = df['destination_ip'].fillna('0.0.0.0')
        mitre_df['DeviceAddress'] = '172.16.254.2'  # Bro sensor IP
        
        # Map hostnames
        mitre_df['SourceHostName'] = df['source_ip'].apply(lambda x: f"host-{str(x).replace('.', '-')}")
        mitre_df['DestinationHostName'] = df['destination_ip'].apply(lambda x: f"target-{str(x).replace('.', '-')}")
        mitre_df['DeviceHostName'] = 'bro-sensor-alpha'
        
        # Map user information (Bro doesn't have user info)
        mitre_df['SourceUserName'] = 'network_service'
        
        # Map timestamps
        mitre_df['EndDate'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(pd.Timestamp.now())
        
        # Map network information
        mitre_df['NetworkProtocol'] = df['protocol']
        mitre_df['SourcePort'] = df['source_port']
        mitre_df['DestinationPort'] = df['destination_port']
        mitre_df['BroFlowId'] = df['bro_uid']
        
        # Map attack information (Bro logs are typically benign unless correlated)
        mitre_df['MalwareIntelAttackType'] = 'Network_Flow'
        mitre_df['AttackSeverity'] = 1  # Low severity for network flows
        mitre_df['Is_Attack'] = 0
        mitre_df['Tactic'] = 'None'
        
        # Generate unique alert IDs
        mitre_df['AlertId'] = 'Bro_' + df.index.astype(str) + '_' + mitre_df['EndDate'].astype(str)
        
        return mitre_df
    
    def stratified_sample(self, ecar_bro_dir: Union[str, Path], 
                         bro_dir: Union[str, Path], 
                         n: int = 1000, 
                         seed: int = 42) -> pd.DataFrame:
        """
        Create a stratified sample of OpTC data maintaining attack distribution.
        
        Args:
            ecar_bro_dir: Directory containing eCAR-Bro JSON files
            bro_dir: Directory containing Bro logs
            n: Total sample size
            seed: Random seed for reproducibility
            
        Returns:
            Stratified sample DataFrame
        """
        np.random.seed(seed)
        
        # Load full dataset
        full_df = self.load_and_preprocess(ecar_bro_dir, bro_dir)
        if full_df.empty:
            return full_df
            
        # Calculate attack type distribution
        attack_counts = full_df['MalwareIntelAttackType'].value_counts()
        attack_props = attack_counts / len(full_df)
        
        # Create stratified sample
        samples = []
        for attack_type, prop in attack_props.items():
            n_samples = max(1, int(np.round(prop * n)))
            subset = full_df[full_df['MalwareIntelAttackType'] == attack_type]
            if len(subset) > 0:
                n_samples = min(n_samples, len(subset))
                samples.append(subset.sample(n=n_samples, random_state=seed))
                
        sampled_df = pd.concat(samples)
        
        # Adjust to exact sample size
        if len(sampled_df) > n:
            sampled_df = sampled_df.sample(n=n, random_state=seed)
        elif len(sampled_df) < n:
            remaining = n - len(sampled_df)
            available = full_df[~full_df.index.isin(sampled_df.index)]
            if len(available) > 0:
                sampled_df = pd.concat([sampled_df, available.sample(n=min(remaining, len(available)), random_state=seed)])
                
        return sampled_df.reset_index(drop=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    loader = DARPAOpTCLoader()
    
    # Set data paths
    optc_base = Path("E:/Private/MITRE-CORE 2/MITRE-CORE_V2/datasets/DARPA_OpTC/OpTCNCR-20260326T025141Z-1-006/OpTCNCR")
    ecar_bro_path = optc_base / "ecar-bro"
    bro_path = optc_base / "bro"
    
    # Load sample data
    try:
        df = loader.stratified_sample(ecar_bro_path, bro_path, n=500)
        print(f"Loaded {len(df)} OpTC records")
        print("\nSample data:")
        print(df[['EndDate', 'SourceAddress', 'DestinationAddress', 'MalwareIntelAttackType', 'Tactic']].head())
        print(f"\nAttack types: {df['MalwareIntelAttackType'].value_counts()}")
        print(f"\nTactics: {df['Tactic'].value_counts()}")
    except Exception as e:
        logger.error(f"Failed to load OpTC data: {e}")
