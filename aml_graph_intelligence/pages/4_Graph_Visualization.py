"""
PAGE 4: Graph Network Visualization
Interactive Multi-Hop Financial Transaction Graph Visualizer
(The Flagship Visualization Module)
"""

import streamlit as st
import streamlit.components.v1 as components
import networkx as nx
import pandas as pd

from utils.helpers import (
    inject_fintech_css, 
    initialize_session_state, 
    render_kpi,
    format_currency
)
from utils.auth import require_authentication, render_officer_sidebar
from visualization.network_visualization import (
    render_pyvis_network, 
    render_plotly_network
)
from graph.graph_analysis import (
    get_k_hop_neighborhood, 
    detect_circular_flows, 
    detect_smurfing_patterns,
    detect_layering_chains,
    detect_suspicious_clusters
)
from models.prediction import generate_risk_reasons

st.set_page_config(
    page_title="Graph Network Visualizer | GraphAML",
    page_icon="🕸️",
    layout="wide"
)

inject_fintech_css()
initialize_session_state()
require_authentication()
render_officer_sidebar()

st.markdown("""
<div style="margin-bottom: 18px;">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <span class="threat-badge threat-normal">FLAGSHIP FORENSIC NETWORK GRAPH</span>
            <h1 style="margin: 6px 0 2px 0;">Interactive Financial Network Intelligence</h1>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
                Examine multi-hop transactional relationships, explore 1-hop / 2-hop / 3-hop ego subgraphs, 
                and uncover structured laundering rings in high-definition physics simulation.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

G = st.session_state.get("G")
df = st.session_state.get("df")
node_df = st.session_state.get("node_df")

if G is None or len(G.nodes) == 0:
    st.warning("⚠️ No graph structure loaded. Please load or upload dataset in Dataset Management.")
    st.stop()

# ==========================================
# 1. GRAPH CONTROL PANEL (SIDEBAR & EXPANDERS)
# ==========================================
all_accounts = sorted(list(G.nodes()))

with st.expander("🛠️ Graph Filter & Exploration Controls", expanded=True):
    gc1, gc2, gc3, gc4 = st.columns(4)
    
    with gc1:
        # Search / Center on account
        search_options = ["-- View Global Graph --"] + all_accounts
        selected_account = st.selectbox(
            "Target Account (Ego-Center)", 
            search_options,
            help="Select an account to extract and highlight its connected multi-hop transaction network."
        )

    with gc2:
        # Hop Depth (1-hop, 2-hop, 3-hop)
        hop_depth = st.selectbox(
            "Neighborhood Hop Depth", 
            [1, 2, 3], 
            index=1,
            help="Select the depth of neighborhood exploration around the target account."
        )

    with gc3:
        # Risk level filter
        risk_filter = st.multiselect(
            "Filter by Risk Levels",
            ["High Risk", "Medium Risk", "Low Risk", "Normal"],
            default=["High Risk", "Medium Risk", "Low Risk", "Normal"]
        )

    with gc4:
        # Rendering Engine & Physics toggle
        engine = st.radio("Graph Engine", ["Interactive Physics (PyVis)", "Plotly 2D WebGL"], horizontal=True)

    # Sub-row for topology presets and node limit
    sub_c1, sub_c2, sub_c3 = st.columns([1.2, 1.2, 1])
    
    with sub_c1:
        topology_preset = st.selectbox(
            "Highlight Forensic Topology Preset",
            ["None / Custom", "Circular Round-Tripping Loops", "Smurfing (Fan-In Aggregators)", "Layering Chains (Multi-Hop)"]
        )

    with sub_c2:
        node_limit = st.slider("Max Display Nodes (Performance Optimization)", min_value=20, max_value=300, value=120, step=10)

    with sub_c3:
        physics_active = st.checkbox("Enable Physics Simulation", value=True)

# ==========================================
# 2. SUBGRAPH EXTRACTION & FILTERING
# ==========================================
highlighted_nodes = []
highlighted_edges = []
target_acc = None

# Case 1: Topology Presets
if topology_preset == "Circular Round-Tripping Loops":
    cycles = detect_circular_flows(G, max_cycle_length=5, limit=5)
    if cycles:
        cycle_nodes = []
        for c in cycles:
            cycle_nodes.extend(c["cycle_path"])
            for u, v, _ in c["edges"]:
                highlighted_edges.append((u, v))
        target_subgraph_nodes = set(cycle_nodes)
        highlighted_nodes = list(target_subgraph_nodes)
        sub_G = G.subgraph(target_subgraph_nodes).copy()
    else:
        st.info("No circular loops found.")
        sub_G = G.copy()

elif topology_preset == "Smurfing (Fan-In Aggregators)":
    smurfs = detect_smurfing_patterns(G, min_in_degree=3)
    if smurfs:
        smurf_nodes = []
        for s in smurfs[:4]:
            smurf_nodes.append(s["aggregator_account"])
            smurf_nodes.extend(s["senders"])
            for snd in s["senders"]:
                highlighted_edges.append((snd, s["aggregator_account"]))
        target_subgraph_nodes = set(smurf_nodes)
        highlighted_nodes = [s["aggregator_account"] for s in smurfs[:4]]
        sub_G = G.subgraph(target_subgraph_nodes).copy()
    else:
        st.info("No smurfing aggregators detected.")
        sub_G = G.copy()

elif topology_preset == "Layering Chains (Multi-Hop)":
    chains = detect_layering_chains(G, min_hops=3, limit=5)
    if chains:
        chain_nodes = []
        for ch in chains:
            p = ch["chain_path"]
            chain_nodes.extend(p)
            for i in range(len(p) - 1):
                highlighted_edges.append((p[i], p[i+1]))
        target_subgraph_nodes = set(chain_nodes)
        highlighted_nodes = list(target_subgraph_nodes)
        sub_G = G.subgraph(target_subgraph_nodes).copy()
    else:
        st.info("No multi-hop layering chains detected.")
        sub_G = G.copy()

# Case 2: Target Account Ego Network
elif selected_account != "-- View Global Graph --":
    target_acc = selected_account
    highlighted_nodes = [target_acc]
    sub_G = get_k_hop_neighborhood(G, center_account=target_acc, k=hop_depth, direction="both")
    if len(sub_G.nodes) == 0:
        sub_G = G.subgraph([target_acc]).copy()

# Case 3: Global View filtered by risk
else:
    # Filter nodes by risk level
    allowed_nodes = [n for n, d in G.nodes(data=True) if d.get("risk_level", "Normal") in risk_filter]
    
    # If too many nodes, sample priority on high/medium risk nodes
    if len(allowed_nodes) > node_limit:
        high_risk_n = [n for n in allowed_nodes if G.nodes[n].get("risk_level") == "High Risk"]
        med_risk_n = [n for n in allowed_nodes if G.nodes[n].get("risk_level") == "Medium Risk"]
        other_n = [n for n in allowed_nodes if n not in high_risk_n and n not in med_risk_n]
        
        selected_nodes = (high_risk_n + med_risk_n + other_n)[:node_limit]
        sub_G = G.subgraph(selected_nodes).copy()
    else:
        sub_G = G.subgraph(allowed_nodes).copy()

# ==========================================
# 3. GRAPH METRICS STRIP
# ==========================================
gm1, gm2, gm3, gm4, gm5 = st.columns(5)
with gm1:
    render_kpi("Visible Nodes", f"{len(sub_G.nodes):,}", "Active accounts", "#38bdf8")
with gm2:
    render_kpi("Visible Edges", f"{len(sub_G.edges):,}", "Financial transfers", "#06b6d4")
with gm3:
    vis_vol = sum(d.get("amount", 0.0) for _, _, d in sub_G.edges(data=True))
    render_kpi("Subnetwork Flow", format_currency(vis_vol), "Aggregated volume", "#10b981")
with gm4:
    high_vis = sum(1 for n, d in sub_G.nodes(data=True) if d.get("risk_level") == "High Risk")
    render_kpi("High Threat Nodes", f"{high_vis}", "In current view", "#ef4444")
with gm5:
    render_kpi("Hop Distance", f"{hop_depth}-Hop" if target_acc else "Global", "Exploration radius", "#f59e0b")

# Legend Row
st.markdown("""
<div style="display: flex; gap: 20px; align-items: center; background: #0f172a; padding: 10px 18px; border-radius: 8px; border: 1px solid #1e293b; margin: 12px 0 16px 0; font-size: 0.85rem;">
    <span style="font-weight: 600; color: #94a3b8;">NODE RISK LEGEND:</span>
    <span style="display: flex; align-items: center; gap: 6px;"><span style="height: 12px; width: 12px; border-radius: 50%; background-color: #ef4444; display: inline-block;"></span> <b style="color: #f87171;">Red: High Risk (&ge; 0.70)</b></span>
    <span style="display: flex; align-items: center; gap: 6px;"><span style="height: 12px; width: 12px; border-radius: 50%; background-color: #f59e0b; display: inline-block;"></span> <b style="color: #fbbf24;">Yellow: Medium Risk (0.40 - 0.69)</b></span>
    <span style="display: flex; align-items: center; gap: 6px;"><span style="height: 12px; width: 12px; border-radius: 50%; background-color: #10b981; display: inline-block;"></span> <b style="color: #34d399;">Green: Low Risk (0.05 - 0.39)</b></span>
    <span style="display: flex; align-items: center; gap: 6px;"><span style="height: 12px; width: 12px; border-radius: 50%; background-color: #38bdf8; display: inline-block;"></span> <b style="color: #38bdf8;">Blue: Normal (&lt; 0.05)</b></span>
    <span style="display: flex; align-items: center; gap: 6px; margin-left: auto;"><span style="height: 12px; width: 12px; border-radius: 50%; background-color: #f43f5e; display: inline-block;"></span> <b style="color: #f43f5e;">Crimson Glow: Selected Ego Center / Suspicious Path</b></span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 4. RENDER GRAPH
# ==========================================
if engine == "Interactive Physics (PyVis)":
    html_graph = render_pyvis_network(
        sub_G,
        height="740px",
        width="100%",
        highlight_nodes=highlighted_nodes,
        highlight_edges=highlighted_edges,
        physics_enabled=physics_active
    )
    components.html(html_graph, height=760, scrolling=False)
else:
    plotly_fig = render_plotly_network(sub_G, highlight_nodes=highlighted_nodes)
    st.plotly_chart(plotly_fig, use_container_width=True)

st.markdown("---")

# ==========================================
# 5. ACCOUNT FORENSIC INSPECTOR & CONNECTIVITY
# ==========================================
st.markdown("### 🔍 Forensic Entity Inspector & Connected Topology")

inspector_acc = target_acc if target_acc else (all_accounts[0] if all_accounts else None)
selected_inspector_acc = st.selectbox(
    "Select Account for Deep Inspection", 
    all_accounts, 
    index=all_accounts.index(inspector_acc) if inspector_acc in all_accounts else 0
)

if selected_inspector_acc and selected_inspector_acc in G:
    node_data = G.nodes[selected_inspector_acc]
    reasons = generate_risk_reasons(selected_inspector_acc, G)

    ins_col1, ins_col2 = st.columns([1, 1.2])

    with ins_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h3 style="margin: 0; color: #38bdf8; font-family: monospace;">{selected_inspector_acc}</h3>
                <span class="threat-badge {'threat-high' if node_data.get('risk_score', 0) >= 0.7 else 'threat-medium'}">
                    {node_data.get('risk_level', 'Normal')} ({node_data.get('risk_score', 0):.2f})
                </span>
            </div>
            <div style="margin-top: 14px; font-size: 0.9rem; line-height: 1.8;">
                &bull; Total Volume: <b>${node_data.get('total_volume', 0):,.2f}</b><br/>
                &bull; Inflow: <b>${node_data.get('total_received', 0):,.2f}</b> ({node_data.get('in_degree', 0)} senders)<br/>
                &bull; Outflow: <b>${node_data.get('total_sent', 0):,.2f}</b> ({node_data.get('out_degree', 0)} receivers)<br/>
                &bull; Rapid Pass-Through Velocity: <b>{node_data.get('rapid_pass_through', 0)} events</b><br/>
                &bull; Structuring Deposits (&lt; $10k): <b>{node_data.get('smurf_count', 0)} events</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 🚨 Regulatory Threat Reason Codes")
        for r in reasons:
            st.markdown(f"&bull; <span style='color: #cbd5e1;'>{r}</span>", unsafe_allow_html=True)

    with ins_col2:
        st.markdown("#### 🔄 Direct Connected Counterparties")
        preds = list(G.predecessors(selected_inspector_acc))
        succs = list(G.successors(selected_inspector_acc))

        tab_in, tab_out = st.tabs([f"Inbound Senders ({len(preds)})", f"Outbound Receivers ({len(succs)})"])

        with tab_in:
            if preds:
                in_rows = []
                for p in preds:
                    edge = G[p][selected_inspector_acc]
                    in_rows.append({
                        "Originating Sender": p,
                        "Amount ($)": float(edge.get("amount", 0.0)),
                        "Txns": edge.get("txn_count", 1),
                        "Risk": G.nodes[p].get("risk_level", "Normal"),
                        "Latest Timestamp": edge.get("latest_timestamp", "N/A")
                    })
                st.dataframe(pd.DataFrame(in_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No incoming transfer counterparties.")

        with tab_out:
            if succs:
                out_rows = []
                for s in succs:
                    edge = G[selected_inspector_acc][s]
                    out_rows.append({
                        "Beneficiary Receiver": s,
                        "Amount ($)": float(edge.get("amount", 0.0)),
                        "Txns": edge.get("txn_count", 1),
                        "Risk": G.nodes[s].get("risk_level", "Normal"),
                        "Latest Timestamp": edge.get("latest_timestamp", "N/A")
                    })
                st.dataframe(pd.DataFrame(out_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No outgoing transfer counterparties.")
