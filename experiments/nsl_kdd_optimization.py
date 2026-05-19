"""
NSL-KDD Optimization Script
Tests enhanced features and campaign grouping for improved ARI performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import time
import hdbscan

# Enhanced NSL-KDD feature set
ENHANCED_NSL_FEATURES = [
    "src_bytes", "dst_bytes", "protocol", "service",
    "duration", "conn_state", "history", "orig_pkts", "resp_pkts"
]

# Campaign grouping strategy
NSL_KDD_CAMPAIGN_GROUPS = {
    1: "normal",
    2: "probe", 
    3: "dos",
    4: "u2r", 5: "u2r", 6: "u2r",      # User-to-root attacks
    7: "r2l", 8: "r2l", 9: "r2l", 10: "r2l",  # Remote-to-local
    11: "privilege", 12: "privilege",        # Privilege escalation
    13: "data_theft", 14: "data_theft", 15: "data_theft",  # Data exfiltration
    16: "backdoor", 17: "backdoor"           # Backdoor attacks
}

def load_enhanced_nsl_kdd():
    """Load NSL-KDD with enhanced features and campaign grouping."""
    try:
        df = pd.read_csv('datasets/nsl_kdd/mitre_format.csv')
        
        # Apply campaign grouping
        df['campaign_group'] = df['campaign_id'].map(NSL_KDD_CAMPAIGN_GROUPS)
        df['campaign_group'] = df['campaign_group'].fillna('other')
        
        # Extract enhanced features
        available_features = [col for col in ENHANCED_NSL_FEATURES if col in df.columns]
        features = df[available_features].copy()
        
        # Handle missing values
        features = features.fillna(0)
        
        # Create labels for grouped campaigns
        labels = LabelEncoder().fit_transform(df['campaign_group'])
        
        print(f"Enhanced NSL-KDD loaded:")
        print(f"  Shape: {features.shape}")
        print(f"  Features: {available_features}")
        print(f"  Campaign groups: {len(np.unique(labels))}")
        print(f"  Class distribution: {dict(pd.Series(labels).value_counts())}")
        
        return features.values, labels, available_features
        
    except Exception as e:
        print(f"Error loading enhanced NSL-KDD: {e}")
        return None, None, None

def test_nsl_kdd_optimization():
    """Test NSL-KDD with enhanced configuration."""
    print("\n🧪 Testing NSL-KDD Optimization...")
    
    # Load enhanced dataset
    features, labels, feature_names = load_enhanced_nsl_kdd()
    if features is None:
        return None
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Test with different clustering methods
    results = []
    n_clusters = len(np.unique(labels))
    
    methods = {
        'K-Means': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        'HDBSCAN': hdbscan.HDBSCAN(min_cluster_size=15),
        'Spectral': SpectralClustering(n_clusters=n_clusters, random_state=42)
    }
    
    for method_name, model in methods.items():
        try:
            start_time = time.time()
            pred_labels = model.fit_predict(features_scaled)
            runtime = time.time() - start_time
            
            # Calculate metrics
            mask = pred_labels != -1
            if mask.sum() > 0:
                ari = adjusted_rand_score(labels[mask], pred_labels[mask])
                nmi = normalized_mutual_info_score(labels[mask], pred_labels[mask])
            else:
                ari = 0.0
                nmi = 0.0
            
            results.append({
                'method': method_name,
                'ari': ari,
                'nmi': nmi,
                'runtime': runtime,
                'coverage': mask.mean()
            })
            
            print(f"  {method_name}: ARI={ari:.3f}, NMI={nmi:.3f}, Coverage={mask.mean():.2f}")
            
        except Exception as e:
            print(f"  {method_name}: Failed - {e}")
    
    return results

if __name__ == "__main__":
    print("=== NSL-KDD OPTIMIZATION TEST ===")
    results = test_nsl_kdd_optimization()
    
    if results:
        print(f"\n✅ Optimization complete!")
        print(f"Best method: {max(results, key=lambda x: x['ari'])['method']} (ARI={max(results, key=lambda x: x['ari'])['ari']:.3f})")
    else:
        print("\n❌ Optimization failed")
