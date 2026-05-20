import os
import sys
import torch
import argparse
import pandas as pd
import logging
from pathlib import Path
from torch_geometric.data import HeteroData

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mitre-core.domain_adaptation")

# Ensure hgnn module can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hgnn.hgnn_correlation import HGNNCorrelationEngine
from hgnn.cross_domain_contrastive import CrossDomainContrastiveLoss
from training.train_on_datasets import PublicDatasetGraphConverter, apply_edge_dropout

def adapt_to_domain(alerts_csv: str, base_checkpoint: str, output_checkpoint: str, epochs: int = 10, batch_size: int = 1000, drop_rate: float = 0.15):
    """
    Run Unsupervised Domain Adaptation (Phase 1 NT-Xent) on unlabeled target alerts.
    """
    logger.info(f"Loading unlabeled alerts from {alerts_csv}")
    df = pd.read_csv(alerts_csv)
    
    converter = PublicDatasetGraphConverter()
    
    # Chunk the dataframe into mini-graphs for batch processing
    graphs = []
    for i in range(0, len(df), batch_size):
        chunk = df.iloc[i:i+batch_size]
        g = converter.convert(chunk)
        if g is not None and 'alert' in g.node_types:
            graphs.append(g)
            
    logger.info(f"Created {len(graphs)} unlabeled graphs for adaptation")
    if not graphs:
        logger.error("No valid graphs created.")
        return
        
    device = torch.device('cpu')  # Force CPU for stability as per main trainer
    
    # Determine alert feature dimension from the first valid graph
    alert_feature_dim = 64
    for g in graphs:
        if 'alert' in g.node_types and g['alert'].x is not None:
            alert_feature_dim = g['alert'].x.shape[1]
            break
            
    # Load base model structure (we don't strictly need the classification head)
    # Using HGNNCorrelationEngine handles checkpoint loading safely
    engine = HGNNCorrelationEngine(
        model_path=base_checkpoint,
        device=str(device),
        use_geometric_confidence=True,
        pure_unsupervised=True
    )
    model = engine.model
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    contrastive_fn = CrossDomainContrastiveLoss(temperature=0.07)
    
    best_loss = float('inf')
    best_state = None
    
    logger.info(f"Starting Unsupervised Adaptation ({epochs} epochs)...")
    for epoch in range(epochs):
        total_loss = 0
        
        for graph in graphs:
            optimizer.zero_grad()
            
            # Create augmented views via edge dropout
            g1 = apply_edge_dropout(graph, drop_rate=drop_rate).to(device)
            g2 = apply_edge_dropout(graph, drop_rate=drop_rate).to(device)
            
            # Pad alert features from 6-dim to 15-dim to match loaded encoder (contextual features removed post-v5)
            for g in [g1, g2]:
                if 'alert' in g.node_types and g['alert'].x.shape[1] == 6:
                    g['alert'].x = torch.nn.functional.pad(g['alert'].x, (0, 9))
            
            # Forward pass
            _, x_dict1 = model(g1)
            _, x_dict2 = model(g2)
            
            if 'alert' in x_dict1 and 'alert' in x_dict2:
                z1 = x_dict1['alert']
                z2 = x_dict2['alert']
                
                # NT-Xent contrastive loss on alert-level embeddings (Phase 1.4 fix)
                loss = contrastive_fn(z1, z2, "target", "target")
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
        avg_loss = total_loss / len(graphs)
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = model.state_dict()
            
    if best_state is not None:
        # Save the adapted backbone
        out_dir = Path(output_checkpoint).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # We save it in the same format expected by the engine
        torch.save({
            'model_state_dict': best_state,
            'loss': best_loss,
            'epoch': epochs
        }, output_checkpoint)
        logger.info(f"Adaptation complete. Adapted checkpoint saved to {output_checkpoint}")
    else:
        logger.error("Adaptation failed to produce a valid model state.")

def main():
    parser = argparse.ArgumentParser(description="MITRE-CORE Unsupervised Domain Adaptation")
    parser.add_argument("input_csv", help="Path to your raw, unlabeled alerts CSV")
    parser.add_argument("--base_checkpoint", default="hgnn_checkpoints/multidomain_v2/best_supervised.pt", help="Base checkpoint to adapt from")
    parser.add_argument("--output_checkpoint", default="hgnn_checkpoints/adapted/best.pt", help="Path to save adapted checkpoint")
    parser.add_argument("--epochs", type=int, default=10, help="Number of adaptation epochs")
    parser.add_argument("--batch_size", type=int, default=1000, help="Alerts per mini-graph")
    
    args = parser.parse_args()
    adapt_to_domain(args.input_csv, args.base_checkpoint, args.output_checkpoint, epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
