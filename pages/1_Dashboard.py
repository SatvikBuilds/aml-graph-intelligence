"""
PAGE 1: Dashboard / Overview
Executive Anti-Money Laundering Command Center
"""

import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import (
    inject_fintech_css, 
    initialize_session_state, 
    render_kpi, 
    format_currency
)
from utils.auth import require_authentication, render_officer_sidebar
from visualization.charts import (
    create_volume_timeline_chart,
    create_suspicious_vs_legit_chart,
    create_risk_distribution_chart,
    create_daily_trend_chart,
    create_top_accounts_bar
)
from graph.graph_analysis import (
    detect_circular_flows,
    detect_smurfing_patterns,
    detect_suspicious_clusters
)

st.set_page_config(
    page_title="Executive AML Dashboard | GraphAML",
    page_icon="📊",
    layout="wide"
)

inject_fintech_css()
initialize_session_state()
require_authentication()
render_officer_sidebar()

# Header
st.markdown("""
<div style="
    padding: 28px 0 20px;
    border-bottom: 1px solid rgba(167,139,250,0.12);
    margin-bottom: 28px;
">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <span style="
                display:inline-flex; align-items:center; gap:6px;
                background:rgba(124,58,237,0.12); border:1px solid rgba(167,139,250,0.3);
                border-radius:999px; padding:4px 12px;
                font-size:0.70rem; font-weight:700; color:#c4b5fd;
                letter-spacing:0.08em; text-transform:uppercase;
                font-family:'DM Mono',monospace;
            ">📊 EXECUTIVE OVERVIEW</span>
            <h1 style="margin:10px 0 4px; font-family:'Sora',sans-serif; letter-spacing:-0.04em;">
                AML Command Center
            </h1>
            <p style="color:#9d8fc4; font-size:0.93rem; margin:0;">
                Macro-level transaction analytics, risk distributions and graph topology signals.
            </p>
        </div>
        <div style="
            background:rgba(26,16,53,0.7); border:1px solid rgba(167,139,250,0.15);
            border-radius:14px; padding:12px 18px; text-align:right;
        ">
            <div style="font-size:0.68rem; color:#6d6a8a; font-weight:600; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:4px;">DATA SOURCE</div>
            <div style="font-size:0.85rem; color:#c4b5fd; font-weight:700;">Synthetic Demo Dataset</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

df = st.session_state.get("df")
node_df = st.session_state.get("node_df")
G = st.session_state.get("G")
preds_df = st.session_state.get("predictions_df")

if df is None or len(df) == 0:
    st.warning("⚠️ No transaction dataset currently active. Please upload a dataset in the Dataset Management module.")
    st.stop()

# Compute KPI Metrics
total_txns = len(df)
total_accounts = len(node_df) if node_df is not None else 0
total_volume = float(df["amount"].sum())

# Suspicious Transactions count
if "is_laundering" in df.columns:
    suspicious_txns = int((df["is_laundering"] == 1).sum())
elif "detection_status" in df.columns:
    suspicious_txns = int((df["detection_status"] == "Suspicious").sum())
else:
    suspicious_txns = 0

# High-Risk Accounts (risk_score >= 0.70)
if preds_df is not None:
    high_risk_accounts = int((preds_df["risk_score"] >= 0.70).sum())
    account_scores = preds_df["risk_score"].tolist()
else:
    high_risk_accounts = 0
    account_scores = [0.0]

# Suspicious Networks Detected (Cycles + Smurf clusters + Connected clusters)
cycles = detect_circular_flows(G, max_cycle_length=5, limit=10) if G is not None else []
smurfs = detect_smurfing_patterns(G, min_in_degree=4) if G is not None else []
clusters = detect_suspicious_clusters(G, high_risk_threshold=0.65) if G is not None else []
suspicious_networks_count = len(cycles) + len(smurfs) + len(clusters)

# ==========================================
# 1. KPI CARDS ROW (6 required KPIs)
# ==========================================
kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

with kpi1:
    render_kpi("Total Transactions", f"{total_txns:,}", "Analyzed in graph", "#a78bfa", icon="💳")
with kpi2:
    render_kpi("Total Accounts", f"{total_accounts:,}", "Entities in network", "#818cf8", icon="🏦")
with kpi3:
    render_kpi("Suspicious Txns", f"{suspicious_txns:,}", f"{(suspicious_txns/total_txns*100):.1f}% of total" if total_txns > 0 else "", "#fb7185", icon="⚠️")
with kpi4:
    render_kpi("High-Risk Accounts", f"{high_risk_accounts}", "Threat score ≥ 0.70", "#f43f5e", icon="🚨")
with kpi5:
    render_kpi("Suspicious Networks", f"{suspicious_networks_count}", f"{len(cycles)} cycles · {len(smurfs)} smurfs", "#fbbf24", icon="🕸️")
with kpi6:
    render_kpi("Total Volume", format_currency(total_volume), "Total currency tracked", "#34d399", icon="💰")

st.markdown("<br/>", unsafe_allow_html=True)

# ==========================================
# 2. CHARTS GRID (2x2)
# ==========================================
c_row1_col1, c_row1_col2 = st.columns(2)

with c_row1_col1:
    fig_vol = create_volume_timeline_chart(df)
    st.plotly_chart(fig_vol, use_container_width=True)

with c_row1_col2:
    fig_pie = create_suspicious_vs_legit_chart(df)
    st.plotly_chart(fig_pie, use_container_width=True)

c_row2_col1, c_row2_col2 = st.columns(2)

with c_row2_col1:
    fig_dist = create_risk_distribution_chart(account_scores)
    st.plotly_chart(fig_dist, use_container_width=True)

with c_row2_col2:
    fig_daily = create_daily_trend_chart(df)
    st.plotly_chart(fig_daily, use_container_width=True)

st.markdown("<br/>", unsafe_allow_html=True)

# ==========================================
# 3. LOWER SECTION: High-Risk Accounts & Recent Suspicious Alerts
# ==========================================
sec_col1, sec_col2 = st.columns([1, 1])

with sec_col1:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px;">
        <span style="font-size:1.1rem;">🚨</span>
        <h3 style="margin:0;font-size:1.0rem;color:#ede9fe;">Top 5 High-Risk Accounts</h3>
    </div>
    """, unsafe_allow_html=True)
    if preds_df is not None:
        top_5 = preds_df.sort_values(by="risk_score", ascending=False).head(5)
        # Display horizontal bar chart
        st.plotly_chart(create_top_accounts_bar(top_5), use_container_width=True)
        
        # Display mini detail table
        st.dataframe(
            top_5[["account_id", "risk_score", "risk_level", "detection_mode"]].rename(columns={
                "account_id": "Account ID",
                "risk_score": "AML Risk Score",
                "risk_level": "Category",
                "detection_mode": "Model Source"
            }),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No prediction data available.")

with sec_col2:
    st.markdown("### 🔔 Latest Investigation Alerts")
    
    # Extract suspicious transactions
    if "is_laundering" in df.columns:
        susp_txns = df[df["is_laundering"] == 1]
    else:
        susp_txns = df[df.get("detection_status", "") == "Suspicious"]
        
    if len(susp_txns) > 0:
        recent_susp = susp_txns.sort_values(by="timestamp", ascending=False).head(5)
        
        for _, alert in recent_susp.iterrows():
            amt = float(alert.get("amount", 0.0))
            sender = alert.get("sender_account", "N/A")
            receiver = alert.get("receiver_account", "N/A")
            txn_id = alert.get("transaction_id", "N/A")
            ts = alert.get("timestamp", "N/A")
            l_type = alert.get("laundering_type", "Rapid Multi-Hop Transfer")
            
            st.markdown(f"""
            <div class="alert-panel" style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 600; color: #f87171; font-family: monospace;">{txn_id} &bull; {l_type}</span>
                    <span style="font-size: 0.8rem; color: #94a3b8;">{ts}</span>
                </div>
                <div style="margin-top: 4px; font-size: 0.88rem; color: #cbd5e1;">
                    Transfer of <b style="color: #38bdf8;">${amt:,.2f}</b> from <code style="color: #f1f5f9;">{sender}</code> &rarr; <code style="color: #f1f5f9;">{receiver}</code>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No active critical transaction alerts detected.")

st.markdown("<br/>", unsafe_allow_html=True)

st.markdown("""
<div style="
    display:flex; align-items:center; gap:10px;
    margin:28px 0 16px; padding-bottom:12px;
    border-bottom:1px solid rgba(167,139,250,0.12);
">
    <span style="width:8px;height:8px;border-radius:50%;background:linear-gradient(135deg,#7c3aed,#ec4899);box-shadow:0 0 8px rgba(124,58,237,0.6);flex-shrink:0;display:inline-block;"></span>
    <h3 style="margin:0; font-size:1.05rem; color:#ede9fe;">Detected Suspicious Graph Topologies</h3>
</div>
""", unsafe_allow_html=True)
topo_col1, topo_col2, topo_col3 = st.columns(3)

with topo_col1:
    st.markdown(f"""
    <div class="kpi-card">
        <h4 style="margin:0 0 6px 0; color:#a78bfa; font-size:0.95rem;">🔄 Circular Round-Tripping</h4>
        <p style="font-size:0.83rem; color:#6d6a8a; margin:0 0 12px 0;">Directed cycles where capital loops back to origin accounts.</p>
        <span class="threat-badge threat-high">{len(cycles)} Closed Loops</span>
    </div>
    """, unsafe_allow_html=True)

with topo_col2:
    st.markdown(f"""
    <div class="kpi-card">
        <h4 style="margin:0 0 6px 0; color:#a78bfa; font-size:0.95rem;">🌪️ Smurfing / Structuring</h4>
        <p style="font-size:0.83rem; color:#6d6a8a; margin:0 0 12px 0;">Fan-in aggregators receiving multiple deposits below threshold.</p>
        <span class="threat-badge threat-high">{len(smurfs)} Aggregator Hubs</span>
    </div>
    """, unsafe_allow_html=True)

with topo_col3:
    st.markdown(f"""
    <div class="kpi-card">
        <h4 style="margin:0 0 6px 0; color:#a78bfa; font-size:0.95rem;">👥 Suspicious Subgraph Clusters</h4>
        <p style="font-size:0.83rem; color:#6d6a8a; margin:0 0 12px 0;">Tightly coupled account components with high fraud density.</p>
        <span class="threat-badge threat-medium">{len(clusters)} Active Clusters</span>
    </div>
    """, unsafe_allow_html=True)
