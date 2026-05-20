"""
UNSW-NB15 label audit — Stage 1 of the per-dataset lifecycle.
Outputs a human-readable report used to write docs/datasets/unsw_nb15/audit.md.
"""
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = Path("datasets/unsw_nb15/mitre_format.csv")

df = pd.read_csv(CSV_PATH)
sha256 = hashlib.sha256(CSV_PATH.read_bytes()).hexdigest()

print("=== UNSW-NB15 MITRE-FORMAT AUDIT ===")
print(f"File: {CSV_PATH}")
print(f"SHA256: {sha256}")
print(f"Total rows: {len(df):,}")
print(f"Columns ({len(df.columns)}): {list(df.columns)}")
print()

# --- Label track value counts ---
for col in ["tactic", "alert_type", "campaign_id", "attack_cat", "stage"]:
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

# --- Candidate label tracks ---
print("=== CANDIDATE LABEL TRACKS ===")
for col in ["tactic", "alert_type", "campaign_id", "attack_cat"]:
    if col not in df.columns:
        continue
    n = df[col].nunique()
    frac_largest = df[col].value_counts(normalize=True).iloc[0]
    print(f"  {col}: {n} classes, largest class = {frac_largest:.1%} of rows")
print()

# --- Cross-tabulations ---
print("=== CROSS-TABULATIONS ===")
if "tactic" in df.columns and "alert_type" in df.columns:
    print("tactic vs alert_type:")
    print(pd.crosstab(df["tactic"], df["alert_type"], margins=True))
    print()

if "campaign_id" in df.columns and "tactic" in df.columns:
    print("campaign_id vs tactic (top 20 campaign_ids):")
    top_cids = df["campaign_id"].value_counts().head(20).index
    sub = df[df["campaign_id"].isin(top_cids)]
    print(pd.crosstab(sub["campaign_id"], sub["tactic"]))
    print()

if "attack_cat" in df.columns and "tactic" in df.columns:
    print("attack_cat vs tactic:")
    print(pd.crosstab(df["attack_cat"], df["tactic"], margins=True))
    print()

# --- Small-class warning ---
print("=== SMALL CLASS WARNING (< 20 rows in full corpus) ===")
for col in ["tactic", "campaign_id", "attack_cat"]:
    if col not in df.columns:
        continue
    vc = df[col].value_counts()
    small = vc[vc < 20]
    if len(small) > 0:
        print(f"  {col}: {len(small)} classes with <20 rows:")
        print(small.to_string())
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
    for col in ["tactic", "alert_type", "campaign_id", "attack_cat"]:
        if col not in df.columns:
            continue
        print(f"  {col}: {sub10k[col].nunique()} unique classes in subset")
        vc_sub = sub10k[col].value_counts()
        small_sub = vc_sub[vc_sub < 5]
        if len(small_sub):
            print(f"    WARNING: {len(small_sub)} classes with <5 rows in subset: {list(small_sub.index)}")
else:
    print(f"Dataset has only {len(df):,} rows — smaller than 10K default sample.")
    print(f"Recommend using full corpus or a smaller sample size.")
print()

print("=== AUDIT COMPLETE ===")
