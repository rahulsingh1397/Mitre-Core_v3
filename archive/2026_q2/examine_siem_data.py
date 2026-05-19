#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys
import os

def install_openpyxl():
    """Install openpyxl package"""
    try:
        import subprocess
        print("Installing openpyxl...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "openpyxl"])
        print("Successfully installed openpyxl")
        return True
    except Exception as e:
        print(f"Failed to install openpyxl: {e}")
        return False

def examine_excel_files():
    """Examine the SIEM Excel files"""
    
    # Try to install openpyxl if not available
    try:
        import openpyxl
        print("openpyxl is available")
    except ImportError:
        if not install_openpyxl():
            print("Cannot proceed without openpyxl")
            return
    
    # Check both files
    files = ['datasets/SQTK_SIEM/preprocessed.xlsx', 'datasets/SQTK_SIEM/SBI_preprocessed_data.xlsx']
    
    for file_path in files:
        print(f"\n{'='*60}")
        print(f"Examining: {file_path}")
        print(f"{'='*60}")
        
        try:
            df = pd.read_excel(file_path)
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            print(f"\nFirst few rows:")
            print(df.head())
            
            print(f"\nData types:")
            print(df.dtypes)
            
            print(f"\nBasic statistics:")
            print(df.describe())
            
            # Check for categorical columns
            cat_cols = df.select_dtypes(include=['object']).columns
            print(f"\nCategorical columns ({len(cat_cols)}):")
            for col in cat_cols:
                unique_vals = df[col].nunique()
                print(f"  {col}: {unique_vals} unique values")
                if unique_vals <= 10:
                    print(f"    Values: {df[col].unique()}")
                else:
                    print(f"    Sample values: {df[col].unique()[:5]}...")
            
            # Check for missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                print(f"\nMissing values:")
                for col, count in missing[missing > 0].items():
                    print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
            else:
                print(f"\nNo missing values found")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    examine_excel_files()
