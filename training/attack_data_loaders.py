"""
MITRE-CORE v3 Data Loaders
==========================
Parses new dataset formats into the MITRE-CORE unified alert schema.
Domains supported:
- windows_sysmon (attack_techniques/, malware/)
- siem_risk (suspicious_behaviour/windows_lolbas_risk/)
- cloud_k8s (cisco_isovalent/)
- nvm_endpoint (cisco_network_visibility_module/)
- network_ids (cisco_secure_firewall_threat_defense/, cisco_secure_access/)
"""

import json
import logging
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger("mitre-core.loaders")

class SysmonXMLLoader:
    """Parses Sysmon XML events into MITRE-CORE schema."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load(self, limit: int = None) -> pd.DataFrame:
        """Load Sysmon XML logs and convert to MITRE-CORE schema."""
        all_events = []
        
        for xml_file in self.data_dir.glob("**/*.log"):
            try:
                logger.info(f"Loading {xml_file}")
                with open(xml_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Some files might have multiple XML documents or malformed XML
                # Try to extract individual Event elements
                import re
                event_pattern = r'<Event[^>]*>.*?</Event>'
                events = re.findall(event_pattern, content, re.DOTALL)
                
                if not events:
                    # Try parsing as single XML document
                    try:
                        tree = ET.fromstring(content)
                        events_xml = tree.findall(".//Event")
                        events = [ET.tostring(event, encoding='unicode') for event in events_xml]
                    except ET.ParseError:
                        logger.warning(f"Could not parse XML from {xml_file}")
                        continue
                
                for event_xml in events:
                    try:
                        event_elem = ET.fromstring(event_xml)
                        event_data = self._parse_sysmon_event(event_elem)
                        if event_data:
                            all_events.append(event_data)
                            
                        if limit and len(all_events) >= limit:
                            break
                    except ET.ParseError:
                        continue
                        
                if limit and len(all_events) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Failed to parse {xml_file}: {e}")
                
        df = pd.DataFrame(all_events)
        logger.info(f"Loaded {len(df)} Sysmon events")
        return df
        
    # Windows Event Log XML namespace
    _NS = "http://schemas.microsoft.com/win/2004/08/events/event"

    def _parse_sysmon_event(self, event_elem: ET.Element) -> Optional[Dict]:
        """Parse a single Sysmon event element."""
        try:
            ns = self._NS
            system = event_elem.find(f"{{{ns}}}System")
            event_data = {}

            # Extract basic event info (with namespace)
            if system is not None:
                tc = system.find(f"{{{ns}}}TimeCreated")
                event_data['timestamp'] = (tc.get("SystemTime", "") if tc is not None
                                           else system.findtext(f"{{{ns}}}TimeCreated", default=""))
                event_data['event_id'] = system.findtext(f"{{{ns}}}EventID", default="")
                event_data['computer'] = system.findtext(f"{{{ns}}}Computer", default="")

            # Extract event data (with namespace)
            event_data_dict = {}
            for data in event_elem.findall(f"{{{ns}}}EventData/{{{ns}}}Data"):
                name = data.get("Name", "")
                value = data.text or ""
                event_data_dict[name] = value
                
            # Map to MITRE-CORE schema
            mapped = {
                'SourceAddress': event_data_dict.get('SourceIp', ''),
                'DestinationAddress': event_data_dict.get('DestinationIp', ''),
                'SourceHostName': event_data.get('computer', ''),
                'DestinationHostName': '',
                'DeviceAddress': event_data.get('computer', ''),
                'DeviceHostName': event_data.get('computer', ''),
                'UserName': event_data_dict.get('User', ''),
                'ProcessName': event_data_dict.get('Image', ''),
                'CommandLine': event_data_dict.get('CommandLine', ''),
                'Timestamp': event_data.get('timestamp', ''),
                'EventID': event_data.get('event_id', ''),
                'Label': 'attack'  # Sysmon events in attack_data are attacks
            }
            
            return mapped
        except Exception as e:
            logger.warning(f"Failed to parse event: {e}")
            return None

class SIEMRiskLoader:
    """Parses Splunk risk KV format."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load(self, limit: int = None) -> pd.DataFrame:
        """Load SIEM risk events and convert to MITRE-CORE schema."""
        all_events = []
        
        for log_file in self.data_dir.glob("**/*.log"):
            try:
                logger.info(f"Loading {log_file}")
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                            
                        event_data = self._parse_risk_event(line)
                        if event_data:
                            all_events.append(event_data)
                            
                        if limit and len(all_events) >= limit:
                            break
                            
                if limit and len(all_events) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Failed to parse {log_file}: {e}")
                
        df = pd.DataFrame(all_events)
        logger.info(f"Loaded {len(df)} SIEM risk events")
        return df
        
    def _parse_risk_event(self, line: str) -> Optional[Dict]:
        """Parse a single Splunk risk KV line."""
        try:
            # Parse key=value pairs
            kv_pairs = {}
            for pair in line.split(','):
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    kv_pairs[key.strip()] = value.strip('"\n ')
                    
            # Map to MITRE-CORE schema
            mapped = {
                'SourceAddress': kv_pairs.get('sa', ''),
                'DestinationAddress': kv_pairs.get('da', ''),
                'SourceHostName': kv_pairs.get('sh', ''),
                'DestinationHostName': kv_pairs.get('dh', ''),
                'DeviceAddress': kv_pairs.get('dest', ''),
                'DeviceHostName': kv_pairs.get('dest', ''),
                'UserName': kv_pairs.get('user', ''),
                'ProcessName': kv_pairs.get('process', ''),
                'CommandLine': kv_pairs.get('process', ''),
                'Timestamp': kv_pairs.get('_time', ''),
                'EventID': kv_pairs.get('search_name', ''),
                'Label': 'attack',
                'RiskScore': kv_pairs.get('risk_score', ''),
                'MITRETechnique': kv_pairs.get('mitre_attack', '')
            }
            
            return mapped
        except Exception as e:
            logger.warning(f"Failed to parse risk event: {e}")
            return None

class KubernetesEBPFLoader:
    """Parses JSON kprobe events."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load(self, limit: int = None) -> pd.DataFrame:
        """Load k8s eBPF events and convert to MITRE-CORE schema."""
        all_events = []
        
        for json_file in self.data_dir.glob("**/*.log"):
            try:
                logger.info(f"Loading {json_file}")
                with open(json_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                            
                        try:
                            event_json = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                            
                        event_data = self._parse_ebpf_event(event_json)
                        if event_data:
                            all_events.append(event_data)
                            
                        if limit and len(all_events) >= limit:
                            break
                            
                if limit and len(all_events) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Failed to parse {json_file}: {e}")
                
        df = pd.DataFrame(all_events)
        logger.info(f"Loaded {len(df)} k8s eBPF events")
        return df
        
    def _parse_ebpf_event(self, event_json: Dict) -> Optional[Dict]:
        """Parse a single eBPF kprobe event."""
        try:
            process_data = event_json.get('process_kprobe', {})
            process = process_data.get('process', {})
            parent = process_data.get('parent', {})
            
            # Extract k8s metadata
            pod_info = process.get('pod', {})
            container_info = process.get('container', {})
            
            # Map to MITRE-CORE schema
            mapped = {
                'SourceAddress': '',
                'DestinationAddress': '',
                'SourceHostName': pod_info.get('name', ''),
                'DestinationHostName': '',
                'DeviceAddress': '',
                'DeviceHostName': pod_info.get('name', ''),
                'UserName': '',
                'ProcessName': process.get('binary', ''),
                'CommandLine': process.get('arguments', ''),
                'Timestamp': process.get('start_time', ''),
                'EventID': 'kprobe',
                'Label': 'attack',
                'ParentProcess': parent.get('binary', ''),
                'ContainerName': container_info.get('name', ''),
                'ContainerImage': container_info.get('image', {}).get('name', ''),
                'PodName': pod_info.get('name', ''),
                'PodNamespace': pod_info.get('namespace', '')
            }
            
            return mapped
        except Exception as e:
            logger.warning(f"Failed to parse eBPF event: {e}")
            return None

class NVMFlowLoader:
    """Parses NVMflow v9 key-value events."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load(self, limit: int = None) -> pd.DataFrame:
        """Load NVM flow events and convert to MITRE-CORE schema."""
        all_events = []
        
        for flow_file in self.data_dir.glob("**/*.log"):
            try:
                logger.info(f"Loading {flow_file}")
                with open(flow_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                            
                        event_data = self._parse_nvm_event(line)
                        if event_data:
                            all_events.append(event_data)
                            
                        if limit and len(all_events) >= limit:
                            break
                            
                if limit and len(all_events) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Failed to parse {flow_file}: {e}")
                
        df = pd.DataFrame(all_events)
        logger.info(f"Loaded {len(df)} NVM flow events")
        return df
        
    def _parse_nvm_event(self, line: str) -> Optional[Dict]:
        """Parse a single NVM flow line."""
        try:
            # Parse key=value pairs with spaces
            kv_pattern = r'(\w+)="([^"]*)"|(\w+)=([^\s]+)'
            kv_pairs = {}
            
            for match in re.finditer(kv_pattern, line):
                if match.group(1):  # quoted value
                    kv_pairs[match.group(1)] = match.group(2)
                else:  # unquoted value
                    kv_pairs[match.group(3)] = match.group(4)
                    
            # Map to MITRE-CORE schema
            mapped = {
                'SourceAddress': kv_pairs.get('sa', ''),
                'DestinationAddress': kv_pairs.get('da', ''),
                'SourceHostName': kv_pairs.get('ds', ''),
                'DestinationHostName': kv_pairs.get('dh', ''),
                'DeviceAddress': kv_pairs.get('ds', ''),
                'DeviceHostName': kv_pairs.get('ds', ''),
                'UserName': kv_pairs.get('pa', ''),
                'ProcessName': kv_pairs.get('pn', ''),
                'CommandLine': kv_pairs.get('parg', ''),
                'Timestamp': kv_pairs.get('fst', ''),
                'EventID': 'nvmflow',
                'Label': 'attack',
                'SourcePort': kv_pairs.get('sp', ''),
                'DestinationPort': kv_pairs.get('dp', ''),
                'Protocol': kv_pairs.get('pr', ''),
                'BytesIn': kv_pairs.get('ibc', ''),
                'BytesOut': kv_pairs.get('obc', '')
            }
            
            return mapped
        except Exception as e:
            logger.warning(f"Failed to parse NVM event: {e}")
            return None

class NetworkFirewallLoader:
    """Parses network firewall and IDS events."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def load(self, limit: int = None) -> pd.DataFrame:
        """Load firewall events and convert to MITRE-CORE schema."""
        all_events = []
        
        for log_file in self.data_dir.glob("**/*.log"):
            try:
                logger.info(f"Loading {log_file}")
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip():
                            continue
                            
                        event_data = self._parse_firewall_event(line)
                        if event_data:
                            all_events.append(event_data)
                            
                        if limit and len(all_events) >= limit:
                            break
                            
                if limit and len(all_events) >= limit:
                    break
            except Exception as e:
                logger.warning(f"Failed to parse {log_file}: {e}")
                
        df = pd.DataFrame(all_events)
        logger.info(f"Loaded {len(df)} firewall events")
        return df
        
    def _parse_firewall_event(self, line: str) -> Optional[Dict]:
        """Parse a single firewall log line."""
        try:
            # Simple regex for typical firewall log format
            # Example: 2025-01-01T12:00:00+00:00 host firewall: TCP 192.168.1.10:12345 -> 10.0.0.1:80 ALLOW
            parts = line.strip().split()
            
            if len(parts) < 6:
                return None
                
            timestamp = parts[0]
            hostname = parts[1]
            protocol = parts[2]
            
            # Parse source and destination
            src_dst = parts[3] if '>' in parts[3] else parts[4]
            if '->' in src_dst:
                src, dst = src_dst.split('->')
                src_ip, src_port = src.split(':')
                dst_ip, dst_port = dst.split(':')
            else:
                return None
                
            action = parts[-1]
            
            # Map to MITRE-CORE schema
            mapped = {
                'SourceAddress': src_ip,
                'DestinationAddress': dst_ip,
                'SourceHostName': hostname,
                'DestinationHostName': '',
                'DeviceAddress': hostname,
                'DeviceHostName': hostname,
                'UserName': '',
                'ProcessName': '',
                'CommandLine': '',
                'Timestamp': timestamp,
                'EventID': 'firewall',
                'Label': 'attack' if action.upper() in ['DENY', 'BLOCK', 'DROP'] else 'benign',
                'SourcePort': src_port,
                'DestinationPort': dst_port,
                'Protocol': protocol,
                'Action': action
            }
            
            return mapped
        except Exception as e:
            logger.warning(f"Failed to parse firewall event: {e}")
            return None

