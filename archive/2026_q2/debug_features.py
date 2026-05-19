#!/usr/bin/env python3
"""Debug feature dimensions to understand the mismatch."""

import torch
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '.')

from hgnn.hgnn_correlation import AlertToGraphConverter

print('=' * 60)
print('DEBUGGING FEATURE DIMENSIONS')
print('=' * 60)

# Load UNSW data
df = pd.read_csv('datasets/unsw_nb15/mitre_format.csv', nrows=1000)
print(f'Data shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')

# Convert to graph
converter = AlertToGraphConverter()
graph = converter.convert(df)

if graph and 'alert' in graph.node_types:
    print(f'Alert features shape: {graph["alert"].x.shape}')
    print(f'First 3 alert features:')
    print(graph["alert"].x[:3])
    
    # Check what features are being generated
    print('\nFeature breakdown:')
    n_alerts = len(df)
    
    # Tactics
    tactics = pd.Categorical(df["tactic"]).codes
    print(f'Tactics: {tactics.shape} - unique: {len(np.unique(tactics))}')
    
    # Alert types
    alert_types = (df["AttackTechnique"] != "").astype(int).values
    print(f'Alert types: {alert_types.shape}')
    
    # Temporal
    ts_values = df["timestamp"]
    dates = pd.to_datetime(ts_values, unit='s', errors='coerce')
    hour = np.nan_to_num(dates.dt.hour.values, nan=0.0)
    dow = np.nan_to_num(dates.dt.dayofweek.values, nan=0.0)
    print(f'Hour: {hour.shape}')
    print(f'Day of week: {dow.shape}')
    
    # Protocol
    protocols = pd.Categorical(df["protocol"]).codes
    print(f'Protocols: {protocols.shape} - unique: {len(np.unique(protocols))}')
    
    # Service
    services = pd.Categorical(df["service"]).codes
    print(f'Services: {services.shape} - unique: {len(np.unique(services))}')
    
    # Check total
    total_features = 6  # tactics + alert_types + hour + dow + protocols + services
    print(f'\nExpected total features: {total_features}')
    print(f'Actual features from converter: {graph["alert"].x.shape[1]}')
    
else:
    print('No alert nodes or features found')
