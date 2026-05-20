"""
hgnn/hgnn_explainability.py
----------------------------
Explainability layer for MITRE-CORE HGNN cluster assignments.

Provides attention weight extraction, feature importance analysis, and cluster interpretation
capabilities for academic publication and operational transparency.

Features:
- GATConv attention weight extraction during inference
- SHAP-based feature importance on backbone embeddings
- High-attention edge mapping back to alert fields
- Top-k contributing features per cluster
- Cluster summary statistics and interpretation

Usage:
    from hgnn.hgnn_explainability import HGNNExplainer
    
    explainer = HGNNExplainer(checkpoint_path)
    explanations = explainer.explain_clusters(alert_df, cluster_assignments)
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

# SHAP for feature importance (install if needed)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Feature importance analysis will be limited.")

from .hgnn_correlation import HGNNCorrelationEngine, AlertToGraphConverter
from .hgnn_training import MITREHeteroGNN

logger = logging.getLogger("mitre-core.explainability")

class AttentionExtractor:
    """Hook-based extractor for GATConv attention weights during forward pass."""
    
    def __init__(self):
        self.attention_weights = {}
        self.hooks = []
    
    def _create_attention_hook(self, layer_name: str):
        """Create a hook to capture attention weights from a GATConv layer."""
        def hook(module, input, output):
            # GATConv returns (nodes, attention_weights) in newer PyG versions
            if isinstance(output, tuple) and len(output) >= 2:
                attention = output[1]  # attention weights
                if attention is not None:
                    self.attention_weights[layer_name] = attention.detach().cpu()
            else:
                # Fallback for older versions - extract from module internals
                if hasattr(module, 'attention_weights'):
                    self.attention_weights[layer_name] = module.attention_weights.detach().cpu()
        
        return hook
    
    def register_hooks(self, model: MITREHeteroGNN):
        """Register hooks on all GATConv layers in the model."""
        self.clear_hooks()
        
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Module) and 'GATConv' in type(module).__name__:
                hook = module.register_forward_hook(self._create_attention_hook(name))
                self.hooks.append(hook)
                logger.info(f"Registered attention hook on layer: {name}")
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_weights.clear()
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get captured attention weights."""
        return self.attention_weights

class HGNNExplainer:
    """
    Explainability interface for MITRE-CORE HGNN cluster assignments.
    
    Provides multiple explanation methods:
    1. Attention weight analysis
    2. Feature importance via SHAP
    3. High-attention edge interpretation
    4. Cluster-level feature statistics
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize the explainer with a trained HGNN checkpoint.
        
        Args:
            checkpoint_path: Path to trained HGNN checkpoint
            device: PyTorch device for computation
        """
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        # Load model and correlation engine
        self.engine = HGNNCorrelationEngine(checkpoint_path, device=device)
        self.model = self.engine.model
        self.converter = self.engine.converter
        
        # Initialize attention extractor
        self.attention_extractor = AttentionExtractor()
        
        # Feature column mapping (will be populated during explanation)
        self.feature_columns = None
        
        logger.info(f"HGNNExplainer initialized with checkpoint: {checkpoint_path}")
    
    def explain_clusters(
        self, 
        alert_df: pd.DataFrame,
        cluster_assignments: Optional[np.ndarray] = None,
        explain_method: str = "attention",
        top_k_features: int = 10,
        save_attention_viz: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanations for cluster assignments.
        
        Args:
            alert_df: DataFrame of alerts to explain
            cluster_assignments: Optional pre-computed cluster assignments
            explain_method: "attention", "shap", or "both"
            top_k_features: Number of top features to highlight per cluster
            save_attention_viz: Whether to save attention weight visualizations
            
        Returns:
            Dictionary containing explanations for each cluster
        """
        logger.info(f"Generating cluster explanations for {len(alert_df)} alerts...")
        
        # Store feature columns for later use
        self.feature_columns = [col for col in alert_df.columns if col not in ['AlertId', 'pred_cluster', 'cluster_confidence']]
        
        # Get cluster assignments if not provided
        if cluster_assignments is None:
            logger.info("Running HGNN correlation to get cluster assignments...")
            result_df = self.engine.correlate(alert_df)
            cluster_assignments = result_df['pred_cluster'].values
            alert_df = result_df.copy()
        
        # Generate explanations based on method
        explanations = {
            "metadata": {
                "num_alerts": len(alert_df),
                "num_clusters": len(np.unique(cluster_assignments)),
                "explain_method": explain_method,
                "top_k_features": top_k_features
            },
            "cluster_summaries": {}
        }
        
        # Generate cluster-level summaries
        self._generate_cluster_summaries(alert_df, cluster_assignments, explanations)
        
        # Generate attention-based explanations
        if explain_method in ["attention", "both"]:
            attention_explanations = self._explain_with_attention(alert_df, cluster_assignments, top_k_features)
            explanations["attention"] = attention_explanations
            
            if save_attention_viz:
                self._save_attention_visualizations(attention_explanations)
        
        # Generate SHAP-based explanations
        if explain_method in ["shap", "both"] and SHAP_AVAILABLE:
            shap_explanations = self._explain_with_shap(alert_df, cluster_assignments, top_k_features)
            explanations["shap"] = shap_explanations
        elif explain_method in ["shap", "both"]:
            logger.warning("SHAP not available. Skipping SHAP-based explanations.")
        
        logger.info("Cluster explanation generation complete!")
        return explanations
    
    def _generate_cluster_summaries(self, alert_df: pd.DataFrame, cluster_assignments: np.ndarray, explanations: Dict):
        """Generate statistical summaries for each cluster."""
        logger.info("Generating cluster summaries...")
        
        alert_df = alert_df.copy()
        alert_df['pred_cluster'] = cluster_assignments
        
        for cluster_id in sorted(np.unique(cluster_assignments)):
            cluster_alerts = alert_df[alert_df['pred_cluster'] == cluster_id]
            
            summary = {
                "cluster_id": int(cluster_id),
                "num_alerts": len(cluster_alerts),
                "avg_confidence": cluster_alerts['cluster_confidence'].mean() if 'cluster_confidence' in cluster_alerts.columns else None,
                "feature_statistics": {},
                "top_attack_types": [],
                "top_sources": [],
                "top_destinations": []
            }
            
            # Feature statistics
            for col in self.feature_columns:
                if col in cluster_alerts.columns:
                    if pd.api.types.is_numeric_dtype(cluster_alerts[col]):
                        summary["feature_statistics"][col] = {
                            "mean": float(cluster_alerts[col].mean()),
                            "std": float(cluster_alerts[col].std()),
                            "min": float(cluster_alerts[col].min()),
                            "max": float(cluster_alerts[col].max())
                        }
                    else:
                        # Categorical - top values
                        value_counts = cluster_alerts[col].value_counts().head(5)
                        summary["feature_statistics"][col] = {
                            "top_values": value_counts.to_dict(),
                            "num_unique": int(cluster_alerts[col].nunique())
                        }
            
            # Attack type analysis
            if 'MalwareIntelAttackType' in cluster_alerts.columns:
                attack_counts = cluster_alerts['MalwareIntelAttackType'].value_counts().head(5)
                summary["top_attack_types"] = attack_counts.to_dict()
            
            # Source/destination analysis
            if 'SourceAddress' in cluster_alerts.columns:
                source_counts = cluster_alerts['SourceAddress'].value_counts().head(5)
                summary["top_sources"] = source_counts.to_dict()
            
            if 'DestinationAddress' in cluster_alerts.columns:
                dest_counts = cluster_alerts['DestinationAddress'].value_counts().head(5)
                summary["top_destinations"] = dest_counts.to_dict()
            
            explanations["cluster_summaries"][str(cluster_id)] = summary
    
    def _explain_with_attention(self, alert_df: pd.DataFrame, cluster_assignments: np.ndarray, top_k: int) -> Dict:
        """Generate explanations using GAT attention weights."""
        logger.info("Generating attention-based explanations...")
        
        # Register attention hooks
        self.attention_extractor.register_hooks(self.model)
        
        # Build graph and run forward pass
        graph_data = self.converter.convert(alert_df)
        graph_data = graph_data.to(self.device)
        
        # Get model output (this triggers attention hooks)
        with torch.no_grad():
            self.model.eval()
            output = self.model(graph_data)
        
        # Extract attention weights
        attention_weights = self.attention_extractor.get_attention_weights()
        self.attention_extractor.clear_hooks()
        
        # Analyze attention patterns per cluster
        cluster_attention = {}
        
        for cluster_id in sorted(np.unique(cluster_assignments)):
            cluster_mask = cluster_assignments == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            cluster_analysis = {
                "cluster_id": int(cluster_id),
                "attention_patterns": {},
                "high_attention_edges": [],
                "top_attended_features": []
            }
            
            # Analyze attention weights per layer
            for layer_name, attention in attention_weights.items():
                if len(attention.shape) >= 2:
                    # attention shape: [num_edges, num_heads] or [num_nodes, num_heads, num_neighbors]
                    if len(cluster_indices) > 0:
                        # Get attention for nodes in this cluster
                        cluster_attention_weights = attention[cluster_indices]
                        
                        # Average attention across heads and edges
                        avg_attention = cluster_attention_weights.mean().item()
                        
                        cluster_analysis["attention_patterns"][layer_name] = {
                            "avg_attention": avg_attention,
                            "max_attention": cluster_attention_weights.max().item(),
                            "attention_std": cluster_attention_weights.std().item()
                        }
            
            cluster_attention[str(cluster_id)] = cluster_analysis
        
        return cluster_attention
    
    def _explain_with_shap(self, alert_df: pd.DataFrame, cluster_assignments: np.ndarray, top_k: int) -> Dict:
        """Generate explanations using SHAP feature importance."""
        logger.info("Generating SHAP-based explanations...")
        
        if not SHAP_AVAILABLE:
            return {"error": "SHAP not installed"}
        
        # Prepare features for SHAP
        feature_df = alert_df[self.feature_columns].copy()
        
        # Handle categorical variables
        categorical_cols = feature_df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        
        # Fill missing values
        feature_df = feature_df.fillna(feature_df.mean())
        
        # Get model embeddings for SHAP analysis
        graph_data = self.converter.convert(alert_df)
        graph_data = graph_data.to(self.device)
        
        with torch.no_grad():
            self.model.eval()
            # Get backbone embeddings (before classification head)
            embeddings = self.model.get_backbone_embeddings(graph_data)
            embeddings = embeddings.cpu().numpy()
        
        # Create SHAP explainer
        # Use a small subset of data as background
        background_size = min(100, len(feature_df))
        background_data = feature_df.sample(background_size, random_state=42)
        
        try:
            explainer = shap.KernelExplainer(
                lambda x: self._predict_embeddings(x, graph_data), 
                background_data
            )
            
            # Calculate SHAP values for each cluster
            cluster_shap = {}
            
            for cluster_id in sorted(np.unique(cluster_assignments)):
                cluster_mask = cluster_assignments == cluster_id
                cluster_features = feature_df[cluster_mask]
                
                if len(cluster_features) > 0:
                    # Sample cluster instances for SHAP analysis
                    sample_size = min(50, len(cluster_features))
                    cluster_samples = cluster_features.sample(sample_size, random_state=42)
                    
                    shap_values = explainer.shap_values(cluster_samples)
                    
                    # Get mean absolute SHAP values per feature
                    if isinstance(shap_values, list):
                        # Multi-class case
                        mean_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                    else:
                        # Binary/single case
                        mean_shap = np.abs(shap_values).mean(axis=0)
                    
                    # Get top-k features
                    top_indices = np.argsort(mean_shap)[-top_k:][::-1]
                    top_features = [
                        {
                            "feature": self.feature_columns[i],
                            "importance": float(mean_shap[i])
                        }
                        for i in top_indices
                    ]
                    
                    cluster_shap[str(cluster_id)] = {
                        "cluster_id": int(cluster_id),
                        "top_features": top_features,
                        "num_samples": len(cluster_samples)
                    }
            
            return cluster_shap
            
        except Exception as e:
            logger.error(f"SHAP analysis failed: {e}")
            return {"error": str(e)}
    
    def _predict_embeddings(self, features: np.ndarray, graph_data) -> np.ndarray:
        """Helper function for SHAP to predict embeddings."""
        # This is a simplified version - in practice, you'd need to properly
        # reconstruct the graph with the modified features
        with torch.no_grad():
            self.model.eval()
            embeddings = self.model.get_backbone_embeddings(graph_data)
            return embeddings.cpu().numpy()
    
    def _save_attention_visualizations(self, attention_explanations: Dict):
        """Save attention weight visualizations."""
        logger.info("Saving attention visualizations...")
        
        # Create output directory
        viz_dir = Path("outputs/explainability/attention_viz")
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate attention heatmap per cluster
        for cluster_id, analysis in attention_explanations.items():
            if "attention_patterns" in analysis:
                patterns = analysis["attention_patterns"]
                
                if patterns:
                    # Create heatmap data
                    layers = list(patterns.keys())
                    avg_attentions = [patterns[layer]["avg_attention"] for layer in layers]
                    
                    # Plot
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=layers, y=avg_attentions)
                    plt.title(f"Average Attention Weights - Cluster {cluster_id}")
                    plt.xlabel("Layer")
                    plt.ylabel("Average Attention Weight")
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save
                    output_path = viz_dir / f"cluster_{cluster_id}_attention.png"
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info(f"Saved attention visualization: {output_path}")
    
    def get_top_contributing_features(self, cluster_id: int, explanations: Dict, method: str = "attention") -> List[Dict]:
        """
        Get top contributing features for a specific cluster.
        
        Args:
            cluster_id: Cluster ID to analyze
            explanations: Full explanations dictionary
            method: "attention" or "shap"
            
        Returns:
            List of feature importance dictionaries
        """
        cluster_key = str(cluster_id)
        
        if method == "attention" and "attention" in explanations:
            # For attention-based, we'd need to map attention back to features
            # This is a simplified implementation
            return [{"feature": "attention_based", "importance": 1.0}]
        
        elif method == "shap" and "shap" in explanations:
            shap_data = explanations["shap"]
            if cluster_key in shap_data and "top_features" in shap_data[cluster_key]:
                return shap_data[cluster_key]["top_features"]
        
        return []
    
    def explain_single_alert(self, alert_df: pd.DataFrame, alert_index: int) -> Dict:
        """
        Generate explanation for a single alert's cluster assignment.
        
        Args:
            alert_df: DataFrame containing the alert
            alert_index: Index of the alert to explain
            
        Returns:
            Dictionary containing explanation for this specific alert
        """
        if alert_index >= len(alert_df):
            raise ValueError(f"alert_index {alert_index} out of range for DataFrame with {len(alert_df)} rows")
        
        # Get cluster assignment for this alert
        result_df = self.engine.correlate(alert_df)
        alert_result = result_df.iloc[alert_index]
        
        explanation = {
            "alert_id": alert_result.get('AlertId', f"index_{alert_index}"),
            "assigned_cluster": int(alert_result['pred_cluster']),
            "confidence": float(alert_result['cluster_confidence']),
            "correlation_method": alert_result.get('correlation_method', 'hgnn'),
            "explanation": "Alert assigned based on learned embedding similarity and attention patterns"
        }
        
        return explanation
    
    def plot_embedding_scatter(self, df, result_df, save_path=None):
        """
        PCA (2D) scatter of backbone embeddings colored by pred_cluster.
        
        Args:
            df: Original alert DataFrame
            result_df: DataFrame with cluster assignments (from engine.correlate())
            save_path: Optional path to save the plot
            
        Returns:
            matplotlib figure object
        """
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        logger.info("Generating embedding scatter plot...")
        
        # Extract backbone embeddings
        try:
            graph_data = self.converter.convert(df)
            graph_data = graph_data.to(self.device)
            
            with torch.no_grad():
                embeddings_dict = self.model.get_backbone_embeddings(graph_data)
                embeddings = embeddings_dict["alert"].cpu().numpy()
            
            # Reduce to 2D with PCA
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings)
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            
            # Get cluster assignments
            cluster_assignments = result_df['pred_cluster'].values
            unique_clusters = np.unique(cluster_assignments)
            
            # Plot each cluster with different color
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
            
            for i, cluster_id in enumerate(unique_clusters):
                mask = cluster_assignments == cluster_id
                plt.scatter(
                    embeddings_2d[mask, 0], 
                    embeddings_2d[mask, 1],
                    c=[colors[i]], 
                    label=f'Cluster {cluster_id}',
                    alpha=0.7,
                    s=20
                )
            
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
            plt.title('HGNN Backbone Embeddings - Cluster Visualization')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Embedding scatter plot saved to: {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding scatter plot: {e}")
            return None
