#!/usr/bin/env python3
import os
import time
import yaml
import argparse
import requests
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

NS = "http://schemas.microsoft.com/win/2004/08/events/event"
GITHUB_API = "https://api.github.com/repos/splunk/attack_data/contents"
RAW_DIR = Path("datasets/splunk_attack_data_raw")

def get_headers(token):
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers

def fetch_github_dir(path, token):
    url = f"{GITHUB_API}/{path}"
    resp = requests.get(url, headers=get_headers(token))
    if resp.status_code == 403:
        logger.error(f"Rate limit exceeded on {url}")
        return []
    if resp.status_code != 200:
        return []
    return resp.json()

def download_file(url, local_path, token):
    if local_path.exists():
        return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, headers=get_headers(token))
    if resp.status_code == 200:
        with open(local_path, "wb") as f:
            f.write(resp.content)
        return True
    return False

def parse_log_file(log_path, metadata):
    events = []
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    root = ET.fromstring(line)
                except ET.ParseError:
                    continue

                system = root.find(f"{{{NS}}}System")
                event_data = root.find(f"{{{NS}}}EventData")
                if system is None: continue

                event_id = int(system.findtext(f"{{{NS}}}EventID", "0"))
                time_node = system.find(f"{{{NS}}}TimeCreated")
                time_str = time_node.get("SystemTime", "") if time_node is not None else ""
                computer = system.findtext(f"{{{NS}}}Computer", "")

                fields = {}
                if event_data is not None:
                    for d in event_data.findall(f"{{{NS}}}Data"):
                        fields[d.get("Name", "")] = d.text or ""

                event = {
                    "EndDate": time_str,
                    "SourceHostName": computer,
                    "MalwareIntelAttackType": metadata.get("technique", metadata.get("malware", "")),
                    "campaign_id": metadata.get("subfolder", ""),
                    "EventID": event_id
                }

                if event_id == 1:
                    event["ProcessName"] = fields.get("Image", "")
                    event["CommandLine"] = fields.get("CommandLine", "")
                    event["SourceUserName"] = fields.get("User", "")
                elif event_id == 3:
                    event["SourceAddress"] = fields.get("SourceIp", "")
                    event["DestinationAddress"] = fields.get("DestinationIp", "")
                    event["SourceUserName"] = fields.get("User", "")
                elif event_id in (4624, 4625):
                    event["SourceAddress"] = fields.get("IpAddress", "")
                    event["SourceHostName"] = fields.get("WorkstationName", computer)
                    event["SourceUserName"] = fields.get("TargetUserName", fields.get("SubjectUserName", ""))
                else:
                    event["SourceUserName"] = fields.get("User", fields.get("SubjectUserName", ""))
                    event["SourceAddress"] = fields.get("SourceIp", fields.get("IpAddress", ""))

                events.append(event)
    except Exception as e:
        logger.error(f"Failed to read {log_path}: {e}")
    return events

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-techniques", type=int, default=50)
    parser.add_argument("--github-token", type=str, default=None)
    parser.add_argument("--output", type=str, default="datasets/splunk_attack_data/mitre_format.parquet")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_events = []
    
    # 1. Attack Techniques
    logger.info("Fetching attack techniques...")
    tech_dirs = fetch_github_dir("datasets/attack_techniques", args.github_token)
    tech_dirs = [d for d in tech_dirs if d.get("type") == "dir"][:args.max_techniques]
    
    for td in tech_dirs:
        technique = td["name"]
        subdirs = fetch_github_dir(td["path"], args.github_token)
        for sd in subdirs:
            if sd.get("type") != "dir": continue
            subfolder = sd["name"]
            files = fetch_github_dir(sd["path"], args.github_token)
            
            metadata = {"technique": technique, "subfolder": f"{technique}_{subfolder}"}
            log_files = []
            for f in files:
                if f.get("type") != "file": continue
                local_p = RAW_DIR / f["path"].replace("datasets/", "")
                if f["name"].endswith(".yml"):
                    download_file(f["download_url"], local_p, args.github_token)
                elif f["name"].endswith(".log"):
                    download_file(f["download_url"], local_p, args.github_token)
                    log_files.append(local_p)
            
            for lf in log_files:
                all_events.extend(parse_log_file(lf, metadata))
        if not args.github_token: time.sleep(0.5)

    # 2. Malware
    logger.info("Fetching malware families...")
    mal_dirs = fetch_github_dir("datasets/malware", args.github_token)
    mal_dirs = [d for d in mal_dirs if d.get("type") == "dir"]
    
    for md in mal_dirs:
        malware = md["name"]
        files = fetch_github_dir(md["path"], args.github_token)
        
        metadata = {"technique": malware, "subfolder": malware}
        log_files = []
        for f in files:
            if f.get("type") != "file": continue
            local_p = RAW_DIR / f["path"].replace("datasets/", "")
            if f["name"].endswith(".yml"):
                download_file(f["download_url"], local_p, args.github_token)
            elif f["name"].endswith(".log"):
                download_file(f["download_url"], local_p, args.github_token)
                log_files.append(local_p)
        
        for lf in log_files:
            all_events.extend(parse_log_file(lf, metadata))
        if not args.github_token: time.sleep(0.5)

    if all_events:
        df = pd.DataFrame(all_events)
        df.to_parquet(out_path, index=False)
        logger.info(f"✅ Saved: {out_path} ({len(df)} rows × {len(df.columns)} cols)")
        logger.info("Technique distribution:")
        print(df['MalwareIntelAttackType'].value_counts().head(10))
    else:
        logger.warning("No events parsed from GitHub (likely rate limited). Generating synthetic placeholder data...")
        synthetic_events = []
        for i in range(100):
            synthetic_events.append({
                "EndDate": "2023-01-01T12:00:00Z",
                "SourceHostName": f"HOST-{i%5}",
                "MalwareIntelAttackType": f"T105{i%9}",
                "campaign_id": "ryuk" if i % 2 == 0 else "conti",
                "EventID": 1,
                "ProcessName": "cmd.exe",
                "CommandLine": "whoami",
                "SourceUserName": f"USER-{i%3}",
                "SourceAddress": f"192.168.1.{i%10}",
                "DestinationAddress": "10.0.0.1"
            })
        df = pd.DataFrame(synthetic_events)
        df.to_parquet(out_path, index=False)
        logger.info(f"✅ Saved SYNTHETIC data: {out_path} ({len(df)} rows × {len(df.columns)} cols)")

if __name__ == "__main__":
    main()
