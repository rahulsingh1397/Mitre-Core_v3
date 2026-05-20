"""
TON_IoT label audit — Stage 1 of the per-dataset lifecycle.
Outputs a human-readable report used to write docs/datasets/ton_iot/audit.md.
"""
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

PARQUET_PATH = Path("datasets/TON_IoT/mitre_format.parquet")

df = pd.read_parquet(PARQUET_PATH)
sha256 = hashlib.sha256(PARQUET_PATH.read_bytes()).hexdigest()

print("=== TON_IoT MITRE-FORMAT AUDIT ===")
print(f"File: {PARQUET_PATH}")
print(f"SHA256: {sha256}")
print(f"Total rows: {len(df):,}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print()

# Schema comparison vs NSL-KDD/UNSW
MITRE_STANDARD_COLS = ['timestamp','src_ip','dst_ip','hostname','username','alert_type',
                       'tactic','campaign_id','attack_cat','protocol','service',
                       'src_bytes','dst_bytes','stage','data_source']
missing = [c for c in MITRE_STANDARD_COLS if c not in df.columns]
extra = [c for c in df.columns if c not in MITRE_STANDARD_COLS]
print(f"=== SCHEMA DIFF vs NSL-KDD/UNSW ===")
print(f"  Missing columns: {missing}")
print(f"  Extra columns:   {extra}")
print()

# --- Label track value counts ---
for col in ["alert_type", "tactic", "campaign_id", "label"]:
    if col not in df.columns:
        print(f"--- {col}: NOT PRESENT ---")
        continue
    n_unique = df[col].nunique()
    n_null = df[col].isna().sum()
    print(f"--- {col} ({n_unique} unique, {n_null} nulls) ---")
    vc = df[col].value_counts(dropna=False)
    print(vc.to_string())
    print()

# --- Null summary ---
print("=== NULL COUNTS PER COLUMN ===")
null_counts = df.isna().sum()
print(null_counts[null_counts > 0].to_string() if null_counts.any() else "No nulls.")
print()

# --- Cross-tabulations ---
print("=== CROSS-TABULATIONS ===")
if "alert_type" in df.columns and "tactic" in df.columns:
    print("alert_type vs tactic:")
    print(pd.crosstab(df["alert_type"], df["tactic"].fillna("Normal"), margins=True))
    print()

if "campaign_id" in df.columns and "alert_type" in df.columns:
    print("campaign_id vs alert_type:")
    print(pd.crosstab(df["campaign_id"], df["alert_type"], margins=True))
    print()

# --- Class sizes ---
print("=== SMALL CLASS WARNING (< 20 rows in full corpus) ===")
for col in ["alert_type", "tactic"]:
    if col not in df.columns:
        continue
    vc = df[col].value_counts(dropna=False)
    small = vc[vc < 20]
    if len(small) > 0:
        print(f"  {col}: {len(small)} classes with <20 rows: {list(small.index)}")
    else:
        print(f"  {col}: no small classes.")
print()

# --- 10K subset simulation ---
print("=== 10K SUBSET SIMULATION (seed 42) ===")
rng = np.random.default_rng(42)
if len(df) >= 10_000:
    idx = rng.choice(len(df), size=10_000, replace=False)
    sub10k = df.iloc[idx]
    print(f"Subset rows: {len(sub10k)}")
    for col in ["alert_type", "tactic", "campaign_id", "label"]:
        if col not in df.columns:
            continue
        print(f"  {col}: {sub10k[col].nunique()} unique classes in subset")
        vc_sub = sub10k[col].value_counts()
        small_sub = vc_sub[vc_sub < 5]
        if len(small_sub):
            print(f"    WARNING: {len(small_sub)} classes with <5 rows: {list(small_sub.index)}")
else:
    print(f"  Dataset smaller than 10K — use full corpus or reduce sample size.")
print()

# --- Graph construction feasibility ---
print("=== GRAPH CONSTRUCTION FEASIBILITY ===")
for col in ["src_ip", "dst_ip", "hostname", "username", "SourceAddress", "SourceHostName"]:
    present = col in df.columns
    print(f"  {col}: {'PRESENT' if present else 'absent'}")
print("  Note: AlertToGraphConverter uses src_ip/dst_ip for IP-based edges")
print("  and falls back gracefully when hostname/username columns are absent.")
print()

print("=== AUDIT COMPLETE ===")
