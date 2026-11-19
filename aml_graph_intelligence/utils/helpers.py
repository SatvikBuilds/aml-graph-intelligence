"""
Helper Utilities and Session State Management
Provides custom financial tech CSS themes, session state synchronization,
and automatic dataset bootstrap.
"""

import os
import streamlit as st
import pandas as pd
import networkx as nx
from typing import Optional, Dict, Any

from preprocessing.data_cleaning import clean_transaction_data
from preprocessing.feature_engineering import extract_node_features
from graph.graph_builder import build_transaction_graph
from models.prediction import predict_baseline_heuristic, score_transactions

def inject_fintech_css():
    """Injects high-end dark navy financial dashboard CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0a0f1d;
        color: #f1f5f9;
    }

    /* Header styling */
    h1, h2, h3, h4 {
        color: #f8fafc;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* Custom KPI Card */
    .kpi-card {
        background: linear-gradient(145deg, #0f172a, #131d31);
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #06b6d4;
    }
    .kpi-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #94a3b8;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #38bdf8;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.02em;
    }
    .kpi-subtext {
        font-size: 0.78rem;
        color: #64748b;
        margin-top: 4px;
    }

    /* Threat badge indicators */
    .threat-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.03em;
        font-family: 'JetBrains Mono', monospace;
    }
    .threat-high {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .threat-medium {
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .threat-low {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .threat-normal {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }

    /* Custom Alert box */
    .alert-panel {
        background: #0f172a;
        border-left: 4px solid #06b6d4;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 16px;
    }

    /* Streamlit tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        color: #94a3b8;
        padding: 8px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        border-color: #06b6d4 !important;
        color: #38bdf8 !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        border: 1px solid #0284c7;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        border-color: #38bdf8;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.4);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a0f1d !important;
        border-right: 1px solid #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)

def render_kpi(title: str, value: str, subtext: str = "", delta_color: str = "#38bdf8"):
    """Renders a styled KPI metric box."""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value" style="color: {delta_color};">{value}</div>
        <div class="kpi-subtext">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)

def get_sample_csv_path() -> str:
    """Finds path to default sample transactions dataset."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "dataset", "sample_transactions.csv")

def initialize_session_state():
    """Ensures all session state variables are instantiated."""
    if "is_initialized" not in st.session_state:
        st.session_state["is_initialized"] = True
        st.session_state["data_source"] = "Synthetic Demo Dataset"
        st.session_state["raw_df"] = None
        st.session_state["df"] = None
        st.session_state["node_df"] = None
        st.session_state["G"] = None
        st.session_state["quality_report"] = None
        st.session_state["trained_model"] = None
        st.session_state["trained_model_name"] = None
        st.session_state["training_history"] = None
        st.session_state["model_metrics"] = None
        st.session_state["predictions_df"] = None
        st.session_state["investigation_cases"] = {}

        # Automatically bootstrap default sample dataset if available
        sample_path = get_sample_csv_path()
        if os.path.exists(sample_path):
            load_and_process_dataset(pd.read_csv(sample_path), source_name="Synthetic Demonstration Dataset")

def load_and_process_dataset(df_raw: pd.DataFrame, mapping: Optional[Dict[str, str]] = None, source_name: str = "Uploaded Dataset"):
    """
    Cleans dataset, extracts node topological features, builds NetworkX graph,
    and runs baseline threat scoring.
    """
    cleaned_df, report = clean_transaction_data(df_raw, mapping=mapping)
    
    # Feature engineering
    node_df, meta = extract_node_features(cleaned_df)
    
    # Graph construction
    G = build_transaction_graph(cleaned_df, node_features_df=node_df)
    
    # Baseline predictions
    pred_df = predict_baseline_heuristic(node_df, G)
    
    # Score transactions
    account_score_dict = dict(zip(pred_df["account_id"], pred_df["risk_score"]))
    scored_txns = score_transactions(cleaned_df, account_score_dict)

    # Store in session state
    st.session_state["raw_df"] = df_raw
    st.session_state["df"] = scored_txns
    st.session_state["node_df"] = node_df
    st.session_state["G"] = G
    st.session_state["quality_report"] = report
    st.session_state["predictions_df"] = pred_df
    st.session_state["data_source"] = source_name
    st.session_state["meta"] = meta
    
    # Initialize investigation cases for high-risk accounts
    high_risk_accs = pred_df[pred_df["risk_score"] >= 0.70]["account_id"].tolist()
    if "investigation_cases" not in st.session_state or not st.session_state["investigation_cases"]:
        cases = {}
        for idx, acc in enumerate(high_risk_accs[:15]):
            status = "New" if idx > 4 else ("Under Investigation" if idx > 1 else "Resolved")
            cases[acc] = {
                "case_id": f"CASE-{2025}-{101 + idx:03d}",
                "account_id": acc,
                "status": status,
                "assigned_officer": f"Officer #0{idx % 4 + 1}",
                "notes": "Automated system flag triggered due to multi-hop AML risk signature."
            }
        st.session_state["investigation_cases"] = cases

def format_currency(val: float) -> str:
    """Formats float as clean USD string."""
    if val >= 1_000_000:
        return f"${val / 1_000_000:.2f}M"
    elif val >= 1_000:
        return f"${val / 1_000:.1f}K"
    else:
        return f"${val:,.2f}"
