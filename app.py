"""
Dynamic Graph Intelligence for Multi-Hop Anti-Money Laundering Detection
Main Application Entry Point
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
import pandas as pd
import os

from utils.helpers import (
    inject_fintech_css, 
    initialize_session_state, 
    format_currency,
    render_kpi
)
from utils.auth import require_authentication, render_officer_sidebar

# Configure Streamlit page
st.set_page_config(
    page_title="GraphAML &bull; Multi-Hop Financial Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark navy fintech theme
inject_fintech_css()

# Initialize session state & default synthetic dataset
initialize_session_state()

# Require Officer Authentication
require_authentication()

# Sidebar Navigation Header
with st.sidebar:
    render_officer_sidebar()
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #1e293b;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: linear-gradient(135deg, #0284c7, #06b6d4); padding: 8px 12px; border-radius: 8px; font-size: 20px;">🕸️</div>
            <div>
                <h3 style="margin: 0; font-size: 1.1rem; color: #f8fafc;">GraphAML Intel</h3>
                <span style="font-size: 0.75rem; color: #38bdf8; letter-spacing: 0.05em;">ENTERPRISE EDITION 2.0</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Active Context")
    df = st.session_state.get("df")
    source = st.session_state.get("data_source", "Demo Data")
    if df is not None:
        st.markdown(f"**Data Source:** `{source}`")
        st.markdown(f"**Transactions:** `{len(df):,}`")
        st.markdown(f"**Unique Accounts:** `{st.session_state.get('node_df', pd.DataFrame()).shape[0]:,}`")
        model_name = st.session_state.get("trained_model_name")
        if model_name:
            st.markdown(f"**Active Model:** <span class='threat-badge threat-high'>{model_name} GNN</span>", unsafe_allow_html=True)
        else:
            st.markdown("**Active Model:** <span class='threat-badge threat-normal'>Baseline Heuristic</span>", unsafe_allow_html=True)
    else:
        st.warning("No dataset loaded.")

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: #64748b; line-height: 1.6;">
        <b>Key Capabilities:</b><br/>
        &bull; Multi-hop Directed Graph Traversal<br/>
        &bull; GCN & GAT Deep Architectures<br/>
        &bull; Circular Round-tripping & Smurfing<br/>
        &bull; Interactive Physics Network (PyVis)<br/>
        &bull; Compliance SAR Dossier Generator
    </div>
    """, unsafe_allow_html=True)

# Main Screen Hero
st.markdown("""
<div style="background: linear-gradient(135deg, rgba(14, 23, 47, 0.95) 0%, rgba(8, 14, 29, 0.98) 100%); border: 1px solid rgba(0, 242, 254, 0.3); border-radius: 20px; padding: 32px; margin-bottom: 24px; box-shadow: 0 20px 50px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255, 255, 255, 0.1);">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <span class="threat-badge threat-normal" style="margin-bottom: 14px;">⚡ MULTI-HOP AML INTELLIGENCE PLATFORM</span>
            <h1 style="font-size: 2.3rem; margin: 10px 0 12px 0; color: #ffffff; font-weight: 800; letter-spacing: -0.03em;">
                Dynamic Graph Intelligence for Multi-Hop Anti-Money Laundering
            </h1>
            <p style="color: #94a3b8; font-size: 1.05rem; max-width: 880px; line-height: 1.6; margin: 0; font-weight: 400;">
                A next-generation financial forensic intelligence platform combining <b>NetworkX graph topology</b>, 
                <b>Graph Convolutional Networks (GCN)</b>, and <b>Graph Attention Networks (GAT)</b> to dismantle 
                sophisticated money laundering topologies, including structuring smurfs, multi-tier layering chains, and circular round-tripping.
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# System Summary KPI Row
if df is not None:
    node_df = st.session_state.get("node_df")
    total_txns = len(df)
    total_accounts = len(node_df) if node_df is not None else 0
    total_vol = float(df["amount"].sum())
    
    # Suspicious accounts count
    preds_df = st.session_state.get("predictions_df")
    if preds_df is not None:
        high_risk_accs = int((preds_df["risk_score"] >= 0.70).sum())
        med_risk_accs = int(((preds_df["risk_score"] >= 0.40) & (preds_df["risk_score"] < 0.70)).sum())
    else:
        high_risk_accs = 0
        med_risk_accs = 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_kpi("Monitored Transactions", f"{total_txns:,}", "Loaded in current session", "#00f2fe", icon="💳")
    with col2:
        render_kpi("Account Entities", f"{total_accounts:,}", "Active graph node vertices", "#4facfe", icon="🏦")
    with col3:
        render_kpi("Total Transaction Flow", format_currency(total_vol), "Cumulative currency tracked", "#00f5a0", icon="💰")
    with col4:
        render_kpi("High-Threat Entities", f"{high_risk_accs}", f"{med_risk_accs} elevated risk accounts", "#ff3366", icon="🚨")

st.markdown("<br/>", unsafe_allow_html=True)

# Navigation Feature Grid
st.markdown("### 🧭 Forensic Navigation Modules")
st.markdown("Select a module from the sidebar or click into the specialized investigation centers below:")

nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    st.markdown("""
    <div class="kpi-card" style="height: 220px;">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">📊</div>
        <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #38bdf8;">1. Executive Dashboard</h3>
        <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
            Real-time executive oversight, financial flow timeline charts, suspicious transaction proportions, and high-threat alert feeds.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/1_Dashboard.py", label="Open Executive Dashboard &rarr;", icon="📊")

with nav_col2:
    st.markdown("""
    <div class="kpi-card" style="height: 220px;">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">📁</div>
        <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #38bdf8;">2. Dataset Management</h3>
        <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
            Ingest custom CSV transaction datasets (e.g. IBM AML or PaySim), map schema headers flexibly, run data quality audits, or load synthetic AML benchmarks.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/2_Dataset_Upload.py", label="Open Dataset Management &rarr;", icon="📁")

with nav_col3:
    st.markdown("""
    <div class="kpi-card" style="height: 220px;">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">🔍</div>
        <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #38bdf8;">3. Transaction Explorer</h3>
        <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
            Searchable, high-performance transaction journal. Filter by risk scoring, amount boundaries, account parties, date ranges, and export records.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/3_Transaction_Explorer.py", label="Open Transaction Explorer &rarr;", icon="🔍")

nav_col4, nav_col5, nav_col6 = st.columns(3)

with nav_col4:
    st.markdown("""
    <div class="kpi-card" style="height: 220px;">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">🕸️</div>
        <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #38bdf8;">4. Graph Network Visualizer</h3>
        <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
            Physics-driven interactive PyVis network. Explore 1-hop, 2-hop, and 3-hop ego neighborhoods, circular round-trips, and suspicious structuring clusters.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/4_Graph_Visualization.py", label="Explore Graph Network &rarr;", icon="🕸️")

with nav_col5:
    st.markdown("""
    <div class="kpi-card" style="height: 220px;">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">🧠</div>
        <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #38bdf8;">5. AI Detection & GNN</h3>
        <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
            Train and evaluate genuine Graph Convolutional (GCN) and Graph Attention (GAT) models. Inspect loss curves, confusion matrices, and ROC-AUC metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/5_AI_Detection.py", label="Launch AI Detection Studio &rarr;", icon="🧠")

with nav_col6:
    st.markdown("""
    <div class="kpi-card" style="height: 220px;">
        <div style="font-size: 1.8rem; margin-bottom: 8px;">🛡️</div>
        <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; color: #38bdf8;">6. Investigation Center</h3>
        <p style="font-size: 0.85rem; color: #94a3b8; line-height: 1.5;">
            Compliance officer workbench: inspect account dossiers, trace source/destination flows, manage case statuses (SAR/STR), and export audit dossiers in HTML/CSV.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/6_Investigation_Center.py", label="Enter Investigation Center &rarr;", icon="🛡️")

st.markdown("<br/>", unsafe_allow_html=True)

# Technical Architecture Overview
with st.expander("🔬 Deep Dive: System Architecture & GNN Technical Pipeline", expanded=False):
    st.markdown("""
    ### System Architecture Breakdown
    
    1. **Graph Construction & Representation (`NetworkX`)**:
       - Accounts are represented as nodes $v \in V$, endowed with structural and transactional features (in/out degree, flow ratios, velocity, PageRank).
       - Transactions are represented as directed edges $e = (u, v) \in E$, attributed with monetary volume, temporal timestamps, and transaction types.
    
    2. **Multi-Hop Topological Pattern Mining**:
       - **Smurfing / Structuring (Fan-In)**: Identifies nodes aggregating multiple deposits under regulatory thresholds ($10,000).
       - **Layering Chains**: Uncovers multi-hop linear pass-through topologies where funds move rapidly between intermediary shell accounts.
       - **Circular Flow / Round-Tripping**: Detects directed cycles $u \to v \to w \to u$ designed to fabricate legitimate commercial provenance.
    
    3. **Graph Neural Network (PyTorch & PyTorch Geometric)**:
       - **GCN Layer Formulation**:
         $$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$$
       - **GAT Multi-Head Attention Formulation**:
         $$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \,\|\, \mathbf{W}h_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \,\|\, \mathbf{W}h_k]\right)\right)}$$
       - Stratified train/val/test splits prevent data leakage across graph topologies.
       - Weighted Cross-Entropy Loss directly counters severe class imbalance (positive AML cases $\\approx 5\\% - 10\\%$).
    
    4. **Explainable AI (XAI)**:
       - Transparent compliance reason codes generated from topological invariants and attention coefficients to satisfy regulatory explainability requirements.
    """)
