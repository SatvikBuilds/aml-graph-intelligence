"""
Feature Engineering Module for Graph-Based AML Detection
Computes node-level graph topological features, temporal flow dynamics,
and pass-through velocity metrics with optimized vectorized NumPy routines.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, Tuple, Any

def extract_node_features(
    df: pd.DataFrame, 
    G: nx.DiGraph = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Extracts rich topological and financial flow features for every account.
    
    Returns:
        node_df: DataFrame where index is account_id with normalized & raw feature vectors
        meta: Metadata dictionary with feature names and scaling parameters
    """
    all_accounts = sorted(list(set(df["sender_account"]).union(set(df["receiver_account"]))))
    
    # 1. Incoming aggregations
    in_agg = df.groupby("receiver_account").agg(
        in_degree=("sender_account", "nunique"),
        in_txn_count=("amount", "count"),
        total_received=("amount", "sum"),
        avg_received=("amount", "mean"),
        max_received=("amount", "max"),
        std_received=("amount", "std")
    ).reset_index().rename(columns={"receiver_account": "account_id"})

    # 2. Outgoing aggregations
    out_agg = df.groupby("sender_account").agg(
        out_degree=("receiver_account", "nunique"),
        out_txn_count=("amount", "count"),
        total_sent=("amount", "sum"),
        avg_sent=("amount", "mean"),
        max_sent=("amount", "max"),
        std_sent=("amount", "std")
    ).reset_index().rename(columns={"sender_account": "account_id"})

    # 3. Structuring / Smurfing indicator: deposits between 8000 and 10000
    structuring_df = df[(df["amount"] >= 8000) & (df["amount"] < 10000)].groupby("receiver_account").size().reset_index(name="smurf_in_count")
    structuring_df = structuring_df.rename(columns={"receiver_account": "account_id"})

    # 4. Rapid Pass-Through Velocity Detection (Optimized NumPy Array Traversal)
    # Fast flow: money received and dispatched within short window (<= 3 hours)
    df_sorted = df.sort_values(by="timestamp").copy()
    df_sorted["ts_dt"] = pd.to_datetime(df_sorted["timestamp"])
    
    in_grouped = {acc: g for acc, g in df_sorted.groupby("receiver_account")}
    out_grouped = {acc: g for acc, g in df_sorted.groupby("sender_account")}
    
    pass_through_counts = {}
    THREE_HOURS_NS = 3 * 3600 * 1_000_000_000  # 3 hours in nanoseconds

    for acc in all_accounts:
        cnt = 0
        if acc in in_grouped and acc in out_grouped:
            in_g = in_grouped[acc]
            out_g = out_grouped[acc]
            
            in_times = in_g["ts_dt"].values.astype(np.int64)
            in_amts = in_g["amount"].values
            out_times = out_g["ts_dt"].values.astype(np.int64)
            out_amts = out_g["amount"].values
            
            for it, ia in zip(in_times, in_amts):
                delta = out_times - it
                # between 0 and 3 hours
                valid_mask = (delta >= 0) & (delta <= THREE_HOURS_NS)
                if np.any(valid_mask):
                    candidate_amts = out_amts[valid_mask]
                    if np.any((candidate_amts >= ia * 0.70) & (candidate_amts <= ia * 1.05)):
                        cnt += 1
                        
        pass_through_counts[acc] = cnt

    pass_through_df = pd.DataFrame([
        {"account_id": acc, "rapid_pass_through_count": cnt} 
        for acc, cnt in pass_through_counts.items()
    ])

    # Merge all into account master
    node_df = pd.DataFrame({"account_id": all_accounts})
    node_df = node_df.merge(in_agg, on="account_id", how="left").fillna(0)
    node_df = node_df.merge(out_agg, on="account_id", how="left").fillna(0)
    node_df = node_df.merge(structuring_df, on="account_id", how="left").fillna(0)
    node_df = node_df.merge(pass_through_df, on="account_id", how="left").fillna(0)

    # Derived flow ratios
    node_df["total_degree"] = node_df["in_degree"] + node_df["out_degree"]
    node_df["total_volume"] = node_df["total_received"] + node_df["total_sent"]
    node_df["net_flow"] = node_df["total_sent"] - node_df["total_received"]
    node_df["in_out_degree_ratio"] = (node_df["in_degree"] + 1) / (node_df["out_degree"] + 1)
    node_df["amount_in_out_ratio"] = (node_df["total_sent"] + 1) / (node_df["total_received"] + 1)

    # 5. Graph Centrality features
    if G is None:
        G = nx.from_pandas_edgelist(
            df, 
            source="sender_account", 
            target="receiver_account", 
            create_using=nx.DiGraph()
        )

    try:
        pagerank_dict = nx.pagerank(G, alpha=0.85, max_iter=200)
    except Exception:
        pagerank_dict = {acc: 1.0 / len(all_accounts) for acc in all_accounts}

    deg_centrality = nx.degree_centrality(G)
    in_centrality = nx.in_degree_centrality(G)
    out_centrality = nx.out_degree_centrality(G)

    node_df["pagerank"] = node_df["account_id"].map(pagerank_dict).fillna(0)
    node_df["degree_centrality"] = node_df["account_id"].map(deg_centrality).fillna(0)
    node_df["in_centrality"] = node_df["account_id"].map(in_centrality).fillna(0)
    node_df["out_centrality"] = node_df["account_id"].map(out_centrality).fillna(0)

    # 6. Target ground truth label (if dataset has labels)
    if "is_laundering" in df.columns:
        suspicious_txns = df[df["is_laundering"] == 1]
        flagged_accounts = set(suspicious_txns["sender_account"]).union(set(suspicious_txns["receiver_account"]))
        node_df["is_suspicious"] = node_df["account_id"].apply(lambda acc: 1 if acc in flagged_accounts else 0)
    else:
        node_df["is_suspicious"] = 0

    # 7. Heuristic Baseline AML Risk Score (0.0 to 1.0)
    norm_pass = np.clip(node_df["rapid_pass_through_count"] / 3.0, 0, 1)
    norm_smurf = np.clip(node_df["smurf_in_count"] / 4.0, 0, 1)
    flow_balance = 1.0 - np.clip(np.abs(node_df["total_sent"] - node_df["total_received"]) / (node_df["total_volume"] + 1e-4), 0, 1)
    is_active = (node_df["total_degree"] >= 2).astype(float)
    
    heuristic_score = (
        0.35 * norm_pass + 
        0.25 * norm_smurf + 
        0.25 * (flow_balance * is_active) + 
        0.15 * np.clip(node_df["degree_centrality"] * 5.0, 0, 1)
    )
    node_df["heuristic_risk_score"] = np.clip(heuristic_score, 0.01, 0.99).round(4)

    feature_cols = [
        "in_degree", "out_degree", "total_degree", "total_received", "total_sent",
        "avg_received", "avg_sent", "in_out_degree_ratio", "amount_in_out_ratio",
        "rapid_pass_through_count", "smurf_in_count", "pagerank", 
        "degree_centrality", "in_centrality", "out_centrality"
    ]

    meta = {
        "all_accounts": all_accounts,
        "feature_cols": feature_cols,
        "num_features": len(feature_cols)
    }

    return node_df, meta
