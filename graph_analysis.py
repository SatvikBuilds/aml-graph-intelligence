"""
Graph Intelligence and Analytics Module
Performs multi-hop neighborhood extraction, bounded circular flow detection,
smurfing/aggregation pattern recognition, and suspicious cluster analysis.
"""

import networkx as nx
import pandas as pd
from typing import List, Dict, Any, Set, Optional

def get_k_hop_neighborhood(
    G: nx.DiGraph, 
    center_account: str, 
    k: int = 2, 
    direction: str = "both"
) -> nx.DiGraph:
    """
    Extracts k-hop ego subgraph around a focal account.
    """
    if center_account not in G:
        return nx.DiGraph()

    visited_nodes: Set[str] = {center_account}
    current_frontier: Set[str] = {center_account}
    node_hop_distance: Dict[str, int] = {center_account: 0}

    for hop in range(1, k + 1):
        next_frontier: Set[str] = set()
        for node in current_frontier:
            if direction in ("out", "both"):
                for succ in G.successors(node):
                    if succ not in visited_nodes:
                        next_frontier.add(succ)
                        visited_nodes.add(succ)
                        node_hop_distance[succ] = hop
            if direction in ("in", "both"):
                for pred in G.predecessors(node):
                    if pred not in visited_nodes:
                        next_frontier.add(pred)
                        visited_nodes.add(pred)
                        node_hop_distance[pred] = hop
        current_frontier = next_frontier
        if not current_frontier:
            break

    sub_G = G.subgraph(visited_nodes).copy()
    for n in sub_G.nodes():
        sub_G.nodes[n]["hop_distance"] = node_hop_distance.get(n, 0)
        if n == center_account:
            sub_G.nodes[n]["is_center"] = True

    return sub_G

def detect_circular_flows(G: nx.DiGraph, max_cycle_length: int = 5, limit: int = 15) -> List[Dict[str, Any]]:
    """
    Finds directed cycles of length 3 to max_cycle_length representing round-tripping.
    Uses bounded DFS traversal for instant execution.
    """
    cycles_found = []
    visited_cycles = set()
    
    nodes = list(G.nodes())
    for start_node in nodes:
        if len(cycles_found) >= limit:
            break
            
        # Stack entries: (current_node, path)
        stack = [(start_node, [start_node])]
        while stack:
            curr, path = stack.pop()
            if len(path) > max_cycle_length:
                continue
                
            for neighbor in G.successors(curr):
                if neighbor == start_node and len(path) >= 3:
                    # Deduplicate cycle rotational symmetry
                    min_idx = path.index(min(path))
                    rot_canon = tuple(path[min_idx:] + path[:min_idx])
                    if rot_canon not in visited_cycles:
                        visited_cycles.add(rot_canon)
                        total_volume = 0.0
                        cycle_edges = []
                        for i in range(len(path)):
                            u = path[i]
                            v = path[(i + 1) % len(path)]
                            amt = G[u][v].get("amount", 0.0)
                            total_volume += amt
                            cycle_edges.append((u, v, amt))
                            
                        cycles_found.append({
                            "cycle_path": path + [path[0]],
                            "length": len(path),
                            "cycle_str": " &rarr; ".join(path) + f" &rarr; {path[0]}",
                            "total_volume": total_volume,
                            "avg_hop_volume": total_volume / len(path),
                            "edges": cycle_edges
                        })
                        if len(cycles_found) >= limit:
                            break
                elif neighbor not in path and len(path) < max_cycle_length:
                    stack.append((neighbor, path + [neighbor]))
                    
            if len(cycles_found) >= limit:
                break

    cycles_found.sort(key=lambda x: x["total_volume"], reverse=True)
    return cycles_found

def detect_smurfing_patterns(
    G: nx.DiGraph, 
    min_in_degree: int = 4, 
    threshold_amount: float = 10000.0
) -> List[Dict[str, Any]]:
    """
    Identifies Smurfing / Structuring Aggregators:
    Accounts receiving funds from multiple distinct sources below reporting threshold.
    """
    aggregators = []
    for node in G.nodes():
        preds = list(G.predecessors(node))
        if len(preds) >= min_in_degree:
            incoming_amounts = [G[pred][node].get("amount", 0.0) for pred in preds]
            under_threshold = [amt for amt in incoming_amounts if amt < threshold_amount]
            if len(under_threshold) >= min_in_degree:
                total_in = sum(incoming_amounts)
                out_deg = G.out_degree(node)
                aggregators.append({
                    "aggregator_account": node,
                    "smurf_count": len(preds),
                    "total_aggregated": total_in,
                    "avg_deposit": total_in / len(preds),
                    "out_degree": out_deg,
                    "senders": preds[:10],
                    "pattern": "Fan-In Aggregator"
                })

    aggregators.sort(key=lambda x: x["smurf_count"], reverse=True)
    return aggregators

def detect_layering_chains(
    G: nx.DiGraph, 
    min_hops: int = 3, 
    max_hops: int = 5, 
    limit: int = 15
) -> List[Dict[str, Any]]:
    """
    Finds multi-hop layering paths (chains where funds flow sequentially).
    """
    chains = []
    start_candidates = [n for n in G.nodes() if G.in_degree(n) <= 1 and G.out_degree(n) >= 1]
    
    def dfs_path(curr, path, depth):
        if depth >= min_hops:
            chains.append(list(path))
            if len(chains) >= limit * 2:
                return
        if depth >= max_hops:
            return
            
        for neighbor in G.successors(curr):
            if neighbor not in path:
                dfs_path(neighbor, path + [neighbor], depth + 1)

    for start_node in start_candidates[:30]:
        dfs_path(start_node, [start_node], 1)
        if len(chains) >= limit:
            break

    formatted_chains = []
    for p in chains[:limit]:
        total_vol = 0.0
        hop_details = []
        for i in range(len(p) - 1):
            amt = G[p[i]][p[i+1]].get("amount", 0.0)
            total_vol += amt
            hop_details.append(f"{p[i]} (${amt:,.0f}) &rarr; {p[i+1]}")
            
        formatted_chains.append({
            "chain_path": p,
            "hop_count": len(p) - 1,
            "chain_str": " &rarr; ".join(p),
            "total_flow": total_vol,
            "hop_details": hop_details
        })

    formatted_chains.sort(key=lambda x: (x["hop_count"], x["total_flow"]), reverse=True)
    return formatted_chains

def detect_suspicious_clusters(G: nx.DiGraph, high_risk_threshold: float = 0.65) -> List[Dict[str, Any]]:
    """
    Identifies connected components containing high-risk nodes and computes risk density.
    """
    U = G.to_undirected()
    components = list(nx.connected_components(U))
    
    suspicious_clusters = []
    for idx, comp in enumerate(components):
        if len(comp) < 2:
            continue
        
        nodes_in_comp = list(comp)
        high_risk_nodes = [n for n in nodes_in_comp if G.nodes[n].get("risk_score", 0.0) >= high_risk_threshold]
        
        if len(high_risk_nodes) > 0:
            cluster_vol = sum(G.nodes[n].get("total_volume", 0.0) for n in nodes_in_comp)
            risk_density = len(high_risk_nodes) / len(nodes_in_comp)
            
            suspicious_clusters.append({
                "cluster_id": f"CLUST_{idx + 1:03d}",
                "total_accounts": len(nodes_in_comp),
                "high_risk_accounts": len(high_risk_nodes),
                "risk_density": risk_density,
                "total_volume": cluster_vol,
                "accounts": nodes_in_comp,
                "high_risk_list": high_risk_nodes
            })

    suspicious_clusters.sort(key=lambda x: (x["high_risk_accounts"], x["risk_density"]), reverse=True)
    return suspicious_clusters

def trace_fund_flow_path(G: nx.DiGraph, source: str, target: str) -> Optional[Dict[str, Any]]:
    """
    Finds the shortest directed path and fund transfer trace between two accounts.
    """
    if source not in G or target not in G:
        return None
    try:
        path = nx.shortest_path(G, source=source, target=target)
        transfers = []
        total_transfer = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = G[u][v]
            amt = edge.get("amount", 0.0)
            total_transfer += amt
            transfers.append({
                "hop": i + 1,
                "from_account": u,
                "to_account": v,
                "amount": amt,
                "timestamp": edge.get("latest_timestamp", "")
            })
        return {
            "path": path,
            "hop_count": len(path) - 1,
            "transfers": transfers,
            "total_path_amount": total_transfer
        }
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
