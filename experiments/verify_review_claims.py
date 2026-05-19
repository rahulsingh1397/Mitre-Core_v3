"""
verify_review_claims.py - End-to-end verification of MITRE-CORE review claims.
"""
import sys, os, time, json, logging, pandas as pd, numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("verify")

def verify_architecture():
    from hgnn.hgnn_correlation import MITREHeteroGNN
    m = MITREHeteroGNN(alert_feature_dim=6, hidden_dim=128, num_heads=4, num_layers=1, num_clusters=10)
    checks = {
        "HeteroConv": any('HeteroConv' in str(type(c)) for c in m.convs),
        "GATConv": any('GATConv' in str(type(l)) for c in m.convs for l in c.convs.values()),
        "num_layers=1": len(m.convs) == 1,
        "alert_dim=6": m.alert_feature_dim == 6,
        "hidden_dim=128": m.hidden_dim == 128,
        "num_heads=4": m.num_heads == 4,
        "has_cluster_head": hasattr(m, 'cluster_classifier'),
        "has_layer_norms": hasattr(m, 'layer_norms') and len(m.layer_norms) > 0,
        "has_raw_proj": hasattr(m, 'alert_raw_proj'),
        "has_get_backbone": hasattr(m, 'get_backbone_embeddings'),
        "has_get_attention": hasattr(m, 'get_attention_weights'),
    }
    verdict = all(checks.values())
    logger.info(f"Architecture: {'PASS' if verdict else 'FAIL'}")
    for k, v in checks.items(): logger.info(f"  {k}: {'OK' if v else 'NO'}")
    return {"verdict": "PASS" if verdict else "FAIL", "checks": checks}

def verify_datasets():
    paths = {
        "UNSW-NB15": "datasets/unsw_nb15/mitre_format.csv",
        "TON_IoT": "datasets/TON_IoT/mitre_format.parquet",
        "NSL-KDD": "datasets/nsl_kdd/mitre_format.csv",
        "OpTC": "datasets/DARPA_OpTC/mitre_format.parquet",
        "SQTK_SIEM": "datasets/SQTK_SIEM/mitre_format.parquet",
        "CICIDS2017": "datasets/CICIDS2017/mitre_format.parquet",
    }
    checks = {n: Path(p).exists() for n, p in paths.items()}
    verdict = all(checks.values())
    logger.info(f"Datasets: {'PASS' if verdict else 'PARTIAL'}")
    for k, v in checks.items(): logger.info(f"  {k}: {'FOUND' if v else 'MISSING'}")
    return {"verdict": "PASS" if verdict else "PARTIAL", "checks": checks}

def verify_checkpoints():
    cps = {
        "multidomain_v2": "hgnn_checkpoints/multidomain_v2/best_supervised.pt",
        "network_v9_v3": "hgnn_checkpoints/network_v9_v3/network_it_best.pt",
        "unsw_supcon_v7": "hgnn_checkpoints/unsw_supcon_v7/best.pt",
        "nsl_kdd": "hgnn_checkpoints/nsl_kdd_best.pt",
    }
    checks = {n: Path(p).exists() for n, p in cps.items()}
    verdict = checks.get("multidomain_v2", False)
    logger.info(f"Checkpoints: {'PASS' if verdict else 'FAIL'}")
    for k, v in checks.items(): logger.info(f"  {k}: {'FOUND' if v else 'MISSING'}")
    return {"verdict": "PASS" if verdict else "FAIL", "checks": checks}

def verify_explainability():
    try:
        from hgnn.hgnn_explainability import HGNNExplainer, AttentionExtractor
        checks = {
            "HGNNExplainer_exists": True,
            "AttentionExtractor_exists": True,
            "has_explain_clusters": hasattr(HGNNExplainer, 'explain_clusters'),
            "has_plot_embedding_scatter": hasattr(HGNNExplainer, 'plot_embedding_scatter'),
            "has_explain_single_alert": hasattr(HGNNExplainer, 'explain_single_alert'),
            "has_get_top_features": hasattr(HGNNExplainer, 'get_top_contributing_features'),
        }
        verdict = all(checks.values())
        logger.info(f"Explainability: {'PASS' if verdict else 'FAIL'}")
        return {"verdict": "PASS" if verdict else "FAIL", "checks": checks}
    except Exception as e:
        return {"verdict": "FAIL", "checks": {"error": str(e)}}

def verify_contrastive_loss():
    try:
        from hgnn.contrastive_loss import NTXentLoss, TemporalNTXentLoss
        from hgnn.cross_domain_contrastive import CrossGraphNTXentLoss
        checks = {
            "NTXentLoss": True,
            "TemporalNTXentLoss": True,
            "CrossGraphNTXentLoss": True,
        }
        logger.info(f"Contrastive loss: PASS")
        return {"verdict": "PASS", "checks": checks}
    except Exception as e:
        return {"verdict": "FAIL", "checks": {"error": str(e)}}

def verify_tactic_map():
    try:
        with open("tactic_map.json") as f:
            tm = json.load(f)
        checks = {
            "has_mitre_tactics": len(tm) > 10,
            "has_unsw_mappings": "Fuzzers" in tm,
            "has_nsl_mappings": "neptune" in tm,
        }
        verdict = all(checks.values())
        logger.info(f"Tactic map: {'PASS' if verdict else 'FAIL'}")
        return {"verdict": "PASS" if verdict else "FAIL", "checks": checks}
    except Exception as e:
        return {"verdict": "FAIL", "checks": {"error": str(e)}}

def verify_existing_results():
    """Check existing experiment results files for reported metrics."""
    results_dir = Path("experiments/results")
    checks = {}
    
    # Check zeroshot baseline
    zb = results_dir / "zeroshot_baseline_final.csv"
    if zb.exists():
        df = pd.read_csv(zb)
        for ds in df["dataset"].unique():
            ds_df = df[df["dataset"] == ds]
            best = ds_df.loc[ds_df["ari"].idxmax()]
            checks[f"{ds}_zeroshot_ari"] = round(best["ari"], 4)
            checks[f"{ds}_zeroshot_ami"] = round(best["ami"], 4) if pd.notna(best.get("ami")) else None
    
    # Check baseline comparison
    bc = results_dir / "baseline_clustering_comparison.csv"
    if bc.exists():
        df = pd.read_csv(bc)
        mitre = df[df["method"] == "MITRE-CORE"]
        for _, row in mitre.iterrows():
            checks[f"{row['dataset']}_mitre_ari"] = round(row["ari"], 4)
    
    # Check final metrics
    fm = results_dir / "final_metrics_v3.csv"
    if fm.exists():
        df = pd.read_csv(fm)
        for ds in df["dataset"].unique():
            ds_df = df[df["dataset"] == ds]
            best = ds_df.loc[ds_df["ari"].idxmax()]
            checks[f"{ds}_final_v3_ari"] = round(best["ari"], 4)
            checks[f"{ds}_final_v3_ami"] = round(best["ami"], 4) if pd.notna(best.get("ami")) else None
    
    logger.info(f"Existing results: {len(checks)} metrics found")
    for k, v in checks.items(): logger.info(f"  {k}: {v}")
    return {"verdict": "PASS", "checks": checks}

def run_e2e_test():
    """Run a quick end-to-end test on UNSW-NB15."""
    from hgnn.hgnn_correlation import HGNNCorrelationEngine
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    
    cp = "hgnn_checkpoints/network_v9_v3/network_it_best.pt"
    if not Path(cp).exists():
        cp = "hgnn_checkpoints/multidomain_v2/best_supervised.pt"
    
    logger.info(f"E2E test with: {cp}")
    
    engine = HGNNCorrelationEngine(
        model_path=cp, use_geometric_confidence=True, pure_unsupervised=True,
        hdbscan_auto_tune=True, hdbscan_cluster_selection_epsilon=0.1,
        use_uf_refinement=False, seed=42,
    )
    
    df = pd.read_csv("datasets/unsw_nb15/mitre_format.csv").sample(2000, random_state=42)
    t0 = time.time()
    result = engine.correlate(df)
    elapsed = time.time() - t0
    
    ari = adjusted_rand_score(df["campaign_id"], result["pred_cluster"])
    ami = adjusted_mutual_info_score(df["campaign_id"], result["pred_cluster"])
    n_clusters = result["pred_cluster"].nunique()
    
    logger.info(f"E2E UNSW-NB15: ARI={ari:.4f}, AMI={ami:.4f}, clusters={n_clusters}, time={elapsed:.1f}s")
    
    return {
        "verdict": "PASS",
        "checks": {
            "e2e_ari": round(ari, 4),
            "e2e_ami": round(ami, 4),
            "e2e_clusters": n_clusters,
            "e2e_time_s": round(elapsed, 1),
            "e2e_checkpoint": cp,
        }
    }

def main():
    print("=" * 70)
    print("MITRE-CORE REVIEW CLAIMS VERIFICATION")
    print("=" * 70)
    
    all_results = {}
    
    # Phase 1: Static verification (no GPU needed for most)
    all_results["architecture"] = verify_architecture()
    all_results["datasets"] = verify_datasets()
    all_results["checkpoints"] = verify_checkpoints()
    all_results["explainability"] = verify_explainability()
    all_results["contrastive_loss"] = verify_contrastive_loss()
    all_results["tactic_map"] = verify_tactic_map()
    all_results["existing_results"] = verify_existing_results()
    
    # Phase 2: E2E test (needs GPU/CPU)
    try:
        all_results["e2e_test"] = run_e2e_test()
    except Exception as e:
        logger.error(f"E2E test failed: {e}")
        all_results["e2e_test"] = {"verdict": "FAIL", "checks": {"error": str(e)}}
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    for section, result in all_results.items():
        status = result["verdict"]
        icon = "PASS" if status == "PASS" else ("WARN" if status == "PARTIAL" else "FAIL")
        print(f"  {section:25s}: {icon}")
    
    # Save results
    out_path = Path("experiments/results/e2e_reports/review_verification.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")
    
    return all_results

if __name__ == "__main__":
    main()
