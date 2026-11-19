"""
Prediction and Explainable AI Inference Module
Generates account-level and transaction-level risk scores using trained GNN models
or clearly-labeled baseline graph heuristics, with compliance reason codes.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Optional, Tuple

from graph.graph_builder import assign_risk_category

def predict_with_gnn(
    model: torch.nn.Module, 
    data, 
    model_name: str = "GCN"
) -> pd.DataFrame:
    """
    Computes genuine GNN inferences on all accounts.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.exp(logits)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    accounts = getattr(data, "accounts", [f"ACC_{i}" for i in range(len(probs))])
    
    results = []
    for i, acc in enumerate(accounts):
        score = float(probs[i])
        risk_level, _ = assign_risk_category(score)
        results.append({
            "account_id": acc,
            "risk_score": round(score, 4),
            "predicted_class": int(preds[i]),
            "risk_level": risk_level,
            "detection_mode": f"GNN ({model_name.upper()})"
        })

    return pd.DataFrame(results)

def predict_baseline_heuristic(node_df: pd.DataFrame, G: nx.DiGraph) -> pd.DataFrame:
    """
    Transparent Rule-Based and Graph Topological Baseline Detector.
    Clearly labeled as baseline heuristic.
    """
    results = []
    for _, row in node_df.iterrows():
        acc = row["account_id"]
        score = float(row.get("heuristic_risk_score", 0.05))
        risk_level, _ = assign_risk_category(score)
        
        results.append({
            "account_id": acc,
            "risk_score": round(score, 4),
            "predicted_class": 1 if score >= 0.50 else 0,
            "risk_level": risk_level,
            "detection_mode": "Baseline Graph Heuristic"
        })

    return pd.DataFrame(results)

def generate_risk_reasons(
    account_id: str, 
    G: nx.DiGraph, 
    node_feat: Optional[pd.Series] = None
) -> List[str]:
    """
    Generates explainable compliance reason codes for flagged accounts.
    """
    reasons = []
    if account_id not in G:
        return ["New or unindexed account entity."]

    node = G.nodes[account_id]
    in_deg = node.get("in_degree", G.in_degree(account_id))
    out_deg = node.get("out_degree", G.out_degree(account_id))
    vol = node.get("total_volume", 0.0)
    score = node.get("risk_score", 0.0)

    # 1. Smurfing / Structuring
    smurf_cnt = node.get("smurf_count", 0)
    if smurf_cnt >= 3:
        reasons.append(f"Structuring / Smurfing: Received {smurf_cnt} transactions just below $10,000 regulatory reporting threshold.")

    # 2. Rapid Velocity / Pass-Through
    pass_cnt = node.get("rapid_pass_through", 0)
    if pass_cnt >= 1:
        reasons.append(f"Rapid Layering Velocity: Dispatched funds in {pass_cnt} instances within 3 hours of receipt.")

    # 3. High Degree Asymmetry (Aggregator vs Disperser)
    if in_deg >= 5 and out_deg <= 1:
        reasons.append(f"Fan-In Aggregator Topology: Funds consolidated from {in_deg} distinct sources into single exit.")
    elif out_deg >= 5 and in_deg <= 1:
        reasons.append(f"Fan-Out Dispersal Topology: Disbursed funds to {out_deg} separate recipient accounts.")

    # 4. High-risk Neighbor Contagion
    high_risk_neighbors = 0
    for neighbor in list(G.predecessors(account_id)) + list(G.successors(account_id)):
        if G.nodes[neighbor].get("risk_score", 0.0) >= 0.70:
            high_risk_neighbors += 1
    if high_risk_neighbors >= 1:
        reasons.append(f"Network Contagion: Direct transfer nexus with {high_risk_neighbors} high-risk AML flagged accounts.")

    # 5. High volume anomaly
    if vol >= 250000:
        reasons.append(f"Elevated Cumulative Volume: Total transacted funds exceed ${vol:,.2f}.")

    if not reasons:
        if score >= 0.50:
            reasons.append("Graph Embedding Anomaly: Multi-hop transaction pattern deviates significantly from legitimate baseline.")
        else:
            reasons.append("Standard activity profile: No high-risk AML indicators observed.")

    return reasons

def score_transactions(
    df: pd.DataFrame, 
    account_scores: Dict[str, float]
) -> pd.DataFrame:
    """
    Computes transaction risk scores based on sender and receiver risk.
    """
    df_scored = df.copy()
    
    def get_txn_risk(row):
        s_score = account_scores.get(row["sender_account"], 0.05)
        r_score = account_scores.get(row["receiver_account"], 0.05)
        # Joint probability / max risk propagation
        base = max(s_score, r_score) * 0.7 + (s_score * r_score) * 0.3
        # Amount weighting
        amt = float(row["amount"])
        if amt >= 8000 and amt < 10000:
            base = min(1.0, base + 0.15)
        elif amt >= 100000:
            base = min(1.0, base + 0.10)
        return round(float(base), 3)

    df_scored["risk_score"] = df_scored.apply(get_txn_risk, axis=1)
    df_scored["detection_status"] = df_scored["risk_score"].apply(
        lambda s: "Suspicious" if s >= 0.65 else ("Elevated Risk" if s >= 0.35 else "Legitimate")
    )
    return df_scored
