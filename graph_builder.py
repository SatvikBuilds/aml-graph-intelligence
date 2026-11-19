"""
Graph Builder Module
Constructs NetworkX directed graphs from transaction records, sets node/edge attributes,
and provides optional Neo4j Cypher generation.
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

def assign_risk_category(score: float) -> Tuple[str, str]:
    """
    Maps risk score [0, 1] to premium financial-intelligence risk label and hex color:
      - High Risk  : >= 0.70 (Rose #f43f5e)
      - Medium Risk: >= 0.40 (Amber #f59e0b)
      - Low Risk   : >= 0.05 (Emerald #10b981)
      - Normal     : <= 0.05 (Violet #7c3aed)
    """
    if score >= 0.70:
        return "High Risk", "#f43f5e"
    elif score >= 0.40:
        return "Medium Risk", "#f59e0b"
    elif score >= 0.05:
        return "Low Risk", "#10b981"
    else:
        return "Normal", "#7c3aed"

def build_transaction_graph(
    df: pd.DataFrame, 
    node_features_df: Optional[pd.DataFrame] = None,
    risk_scores: Optional[Dict[str, float]] = None
) -> nx.DiGraph:
    """
    Constructs an attributed NetworkX DiGraph from transaction data.
    """
    G = nx.DiGraph()

    # Pre-map node features if available
    features_lookup = {}
    if node_features_df is not None:
        features_lookup = node_features_df.set_index("account_id").to_dict(orient="index")

    # Add all unique accounts as nodes
    all_accounts = set(df["sender_account"]).union(set(df["receiver_account"]))
    for acc in all_accounts:
        feat = features_lookup.get(acc, {})
        # Score priority: risk_scores dict > feat['gnn_risk_score'] > feat['heuristic_risk_score'] > 0.0
        score = 0.0
        if risk_scores and acc in risk_scores:
            score = float(risk_scores[acc])
        elif "gnn_risk_score" in feat:
            score = float(feat["gnn_risk_score"])
        elif "heuristic_risk_score" in feat:
            score = float(feat["heuristic_risk_score"])
        
        risk_level, color = assign_risk_category(score)
        
        total_vol = float(feat.get("total_volume", 0.0))
        in_deg = int(feat.get("in_degree", 0))
        out_deg = int(feat.get("out_degree", 0))

        # Size scaled smoothly between 14 and 45
        size = 14 + int(np.clip(np.log1p(total_vol) * 1.8, 0, 30))

        G.add_node(
            acc,
            label=acc,
            account_id=acc,
            risk_score=score,
            risk_level=risk_level,
            color=color,
            size=size,
            total_volume=total_vol,
            total_sent=float(feat.get("total_sent", 0.0)),
            total_received=float(feat.get("total_received", 0.0)),
            in_degree=in_deg,
            out_degree=out_deg,
            is_suspicious=int(feat.get("is_suspicious", 0)),
            rapid_pass_through=int(feat.get("rapid_pass_through_count", 0)),
            smurf_count=int(feat.get("smurf_in_count", 0)),
            title=(
                f"<b>Account:</b> {acc}<br>"
                f"<b>Risk Level:</b> {risk_level} ({score:.2f})<br>"
                f"<b>Total Volume:</b> ${total_vol:,.2f}<br>"
                f"<b>In-Degree:</b> {in_deg} | <b>Out-Degree:</b> {out_deg}"
            )
        )

    # Add transactions as directed edges
    # For multiple transactions between same pair, aggregate volume and keep transaction details
    edge_groups = df.groupby(["sender_account", "receiver_account"])
    for (src, dst), group in edge_groups:
        total_amt = float(group["amount"].sum())
        count = len(group)
        is_suspicious_edge = int((group.get("is_laundering", pd.Series([0])) == 1).any())
        latest_time = str(group["timestamp"].max())
        txn_types = list(group["transaction_type"].unique())
        txn_ids = list(group["transaction_id"].unique())

        # Edge width scaled between 1 and 7
        width = 1.0 + min(float(np.log1p(total_amt) / 2.0), 6.0)
        edge_color = "#ef4444" if is_suspicious_edge else "#475569"

        G.add_edge(
            src,
            dst,
            amount=total_amt,
            txn_count=count,
            latest_timestamp=latest_time,
            is_suspicious=is_suspicious_edge,
            color=edge_color,
            width=width,
            transaction_types=",".join(txn_types),
            transaction_ids=txn_ids[:5], # sample IDs
            title=(
                f"<b>Transfer:</b> {src} &rarr; {dst}<br>"
                f"<b>Aggregated Amount:</b> ${total_amt:,.2f}<br>"
                f"<b>Txn Count:</b> {count}<br>"
                f"<b>Latest:</b> {latest_time}<br>"
                f"<b>Status:</b> {'FLAGGED SUSPICIOUS' if is_suspicious_edge else 'Legitimate'}"
            )
        )

    return G

def export_to_neo4j_cypher(df: pd.DataFrame, max_rows: int = 500) -> str:
    """
    Generates ready-to-run Cypher queries for importing into Neo4j graph database.
    """
    lines = [
        "// === NEO4J CYPHER AML IMPORT SCRIPT ===",
        "// Create constraint for Account node",
        "CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE;",
        "",
        "// Create Account Nodes & Transaction Relationships"
    ]

    sample_df = df.head(max_rows)
    for _, row in sample_df.iterrows():
        src = row["sender_account"]
        dst = row["receiver_account"]
        amt = float(row["amount"])
        ts = str(row["timestamp"])
        tx_type = str(row.get("transaction_type", "TRANSFER"))
        tx_id = str(row.get("transaction_id", "TXN"))
        is_fraud = int(row.get("is_laundering", 0))

        cypher = (
            f"MERGE (s:Account {{id: '{src}'}}) "
            f"MERGE (r:Account {{id: '{dst}'}}) "
            f"CREATE (s)-[:TRANSFERRED {{"
            f"txn_id: '{tx_id}', "
            f"amount: {amt}, "
            f"timestamp: '{ts}', "
            f"type: '{tx_type}', "
            f"is_laundering: {is_fraud}"
            f"}}]->(r);"
        )
        lines.append(cypher)

    lines.append("")
    lines.append("// Cypher query to detect 3-hop money laundering chains:")
    lines.append("// MATCH path = (a:Account)-[:TRANSFERRED]->(b:Account)-[:TRANSFERRED]->(c:Account)-[:TRANSFERRED]->(d:Account)")
    lines.append("// WHERE a <> d RETURN path LIMIT 25;")
    lines.append("")
    lines.append("// Cypher query to detect circular round-tripping:")
    lines.append("// MATCH cycle = (a:Account)-[:TRANSFERRED*3..5]->(a:Account) RETURN cycle LIMIT 10;")
    
    return "\n".join(lines)
