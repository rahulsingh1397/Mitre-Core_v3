"""
CICIDS2017 label audit — Stage 1 of the per-dataset lifecycle.
Outputs a human-readable report used to write docs/datasets/cicids2017/audit.md.
"""
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

PARQUET_PATH = Path("datasets/CICIDS2017/mitre_format.parquet")

df = pd.read_parquet(PARQUET_PATH)
sha256 = hashlib.sha256(PARQUET_PATH.read_bytes()).hexdigest()

print("=== CICIDS2017 MITRE-FORMAT AUDIT ===")
print(f"File: {PARQUET_PATH}")
print(f"SHA256: {sha256}")
print(f"Total rows: {len(df):,}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print()

# Schema comparison vs NSL-KDD/UNSW standard
MITRE_STANDARD_COLS = ['timestamp', 'src_ip', 'dst_ip', 'hostname', 'username',
                       'alert_type', 'tactic', 'campaign_id', 'attack_cat',
                       'protocol', 'service', 'src_bytes', 'dst_bytes',
                       'stage', 'data_source']
missing = [c for c in MITRE_STANDARD_COLS if c not in df.columns]
extra = [c for c in df.columns if c not in MITRE_STANDARD_COLS]
print("=== SCHEMA DIFF vs NSL-KDD/UNSW ===")
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
    print(vc.to_string().encode('ascii', errors='replace').decode('ascii'))
    print()

# --- Null summary ---
print("=== NULL COUNTS PER COLUMN ===")
null_counts = df.isna().sum()
has_nulls = null_counts[null_counts > 0]
print(has_nulls.to_string() if len(has_nulls) > 0 else "No nulls.")
print()

# --- Cross-tabulations ---
print("=== CROSS-TABULATIONS ===")
if "alert_type" in df.columns and "tactic" in df.columns:
    print("alert_type vs tactic (alert_type rows, tactic cols):")
    ct = pd.crosstab(df["alert_type"].fillna("_NULL_"), df["tactic"].fillna("_NULL_"), margins=True)
    print(ct.to_string().encode('ascii', errors='replace').decode('ascii'))
    print()

# --- Small class warnings ---
print("=== SMALL CLASS WARNING ===")
print("  Threshold: <20 rows in full corpus, <5 rows in 10K stratified subset")
for col in ["alert_type", "tactic"]:
    if col not in df.columns:
        continue
    vc = df[col].value_counts(dropna=False)
    small_corpus = vc[vc < 20]
    # Estimate 10K stratified count: proportional
    small_10k = vc[vc * 10_000 / len(df) < 1.0]  # <1 expected row in 10K
    if len(small_corpus) > 0:
        safe_idx = [str(x).encode('ascii', errors='replace').decode('ascii') for x in small_corpus.index]
        print(f"  {col}: {len(small_corpus)} classes with <20 rows: {safe_idx}")
    if len(small_10k) > 0:
        print(f"  {col}: {len(small_10k)} classes with <1 EXPECTED row in 10K subset:")
        for cls, cnt in small_10k.items():
            expected = cnt * 10_000 / len(df)
            cls_safe = str(cls).encode('ascii', errors='replace').decode('ascii')
            print(f"    '{cls_safe}': {cnt} rows in corpus -> {expected:.2f} expected in 10K")
    if len(small_corpus) == 0 and len(small_10k) == 0:
        print(f"  {col}: no small classes.")
print()

# --- Stratified 10K subset simulation (mirrors benchmark stratified_sample=true) ---
print("=== 10K STRATIFIED SUBSET SIMULATION (seed 42, stratified by alert_type) ===")
strat_col = "alert_type"
# sklearn train_test_split with stratify mirrors the benchmark's stratified sampling
try:
    _, sub10k_idx = train_test_split(
        np.arange(len(df)),
        test_size=10_000,
        random_state=42,
        stratify=df[strat_col].values
    )
    sub10k = df.iloc[sub10k_idx]
    print(f"Subset rows: {len(sub10k)}")
    for col in ["alert_type", "tactic", "campaign_id", "label"]:
        if col not in df.columns:
            continue
        vc_sub = sub10k[col].value_counts(dropna=False)
        print(f"\n  {col} ({vc_sub.shape[0]} classes in subset):")
        print(vc_sub.to_string())
        small_sub = vc_sub[vc_sub < 5]
        if len(small_sub) > 0:
            print(f"  WARNING: {len(small_sub)} classes with <5 rows: {list(small_sub.index)}")
except Exception as e:
    # Fallback: classes too sparse for stratified split — use random sample
    print(f"  Stratified split failed ({e}); falling back to random sample.")
    # Drop classes with <2 members before stratification
    vc = df[strat_col].value_counts()
    mask = df[strat_col].map(vc) >= 2
    df_filt = df[mask]
    _, sub10k_idx = train_test_split(
        np.arange(len(df_filt)),
        test_size=min(10_000, len(df_filt)),
        random_state=42,
        stratify=df_filt[strat_col].values
    )
    sub10k = df_filt.iloc[sub10k_idx]
    print(f"  Subset rows (after dropping singleton classes): {len(sub10k)}")
    for col in ["alert_type", "tactic"]:
        if col not in df.columns:
            continue
        vc_sub = sub10k[col].value_counts()
        print(f"  {col}: {vc_sub.shape[0]} classes in subset")
print()

# --- Graph construction feasibility ---
print("=== GRAPH CONSTRUCTION FEASIBILITY ===")
for col in ["src_ip", "dst_ip", "hostname", "username", "SourceAddress", "SourceHostName"]:
    present = col in df.columns
    print(f"  {col}: {'PRESENT' if present else 'absent'}")
print("  Note: AlertToGraphConverter uses src_ip/dst_ip for IP-based edges")
print("  and falls back gracefully when hostname/username are absent.")
print()

print("=== AUDIT COMPLETE ===")
