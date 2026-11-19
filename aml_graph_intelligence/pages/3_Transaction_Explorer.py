"""
PAGE 3: Transaction Explorer
Searchable and Filterable Financial Transaction Ledger with Entity Inspection
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from utils.helpers import (
    inject_fintech_css, 
    initialize_session_state, 
    render_kpi,
    format_currency
)
from utils.auth import require_authentication, render_officer_sidebar
from models.prediction import generate_risk_reasons

st.set_page_config(
    page_title="Transaction Explorer | GraphAML",
    page_icon="🔍",
    layout="wide"
)

inject_fintech_css()
initialize_session_state()
require_authentication()
render_officer_sidebar()

st.markdown("""
<div style="margin-bottom: 20px;">
    <span class="threat-badge threat-normal">TRANSACTION SURVEILLANCE & AUDIT</span>
    <h1 style="margin: 6px 0 2px 0;">Transaction Explorer</h1>
    <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
        Filter, inspect, and trace individual transaction records with multi-dimensional risk scores and entity forensics.
    </p>
</div>
""", unsafe_allow_html=True)

df = st.session_state.get("df")
G = st.session_state.get("G")

if df is None or len(df) == 0:
    st.warning("⚠️ No active transaction dataset. Please load or upload data in Dataset Management.")
    st.stop()

# ==========================================
# 1. FILTER CONTROLS BAR
# ==========================================
with st.expander("🔍 Transaction Filter Controls", expanded=True):
    f_col1, f_col2, f_col3, f_col4 = st.columns(4)

    with f_col1:
        # Account search
        search_account = st.text_input("Search Account ID", placeholder="e.g. ACC_MULE_AGG_01", value="")

    with f_col2:
        # Detection Status Filter
        status_options = ["All Records", "Suspicious Only", "Elevated Risk Only", "Legitimate Only"]
        filter_status = st.selectbox("Detection Status", status_options)

    with f_col3:
        # Risk Score Threshold
        min_risk = st.slider("Minimum Risk Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    with f_col4:
        # Transaction Type
        types_available = ["All Types"] + sorted(list(df["transaction_type"].dropna().unique()))
        filter_type = st.selectbox("Transaction Type", types_available)

    # Date & Amount Filters Row
    f_sub1, f_sub2, f_sub3 = st.columns([1, 1, 1])
    
    with f_sub1:
        min_amt = float(df["amount"].min())
        max_amt = float(df["amount"].max())
        amt_range = st.slider(
            "Transaction Amount Range ($)", 
            min_value=min_amt, 
            max_value=max_amt, 
            value=(min_amt, max_amt)
        )

    with f_sub2:
        # Dates
        df_dates = pd.to_datetime(df["timestamp"]).dt.date
        min_date = df_dates.min()
        max_date = df_dates.max()
        selected_dates = st.date_input("Date Range", value=(min_date, max_date))

    with f_sub3:
        max_rows = st.select_slider("Display Limit (Rows)", options=[50, 100, 250, 500, 1000], value=100)

# Apply filters
filtered_df = df.copy()

# 1. Account Search
if search_account.strip():
    term = search_account.strip().lower()
    filtered_df = filtered_df[
        filtered_df["sender_account"].str.lower().str.contains(term) | 
        filtered_df["receiver_account"].str.lower().str.contains(term)
    ]

# 2. Status Filter
if filter_status == "Suspicious Only":
    filtered_df = filtered_df[filtered_df["detection_status"] == "Suspicious"]
elif filter_status == "Elevated Risk Only":
    filtered_df = filtered_df[filtered_df["detection_status"] == "Elevated Risk"]
elif filter_status == "Legitimate Only":
    filtered_df = filtered_df[filtered_df["detection_status"] == "Legitimate"]

# 3. Risk Score
filtered_df = filtered_df[filtered_df["risk_score"] >= min_risk]

# 4. Type
if filter_type != "All Types":
    filtered_df = filtered_df[filtered_df["transaction_type"] == filter_type]

# 5. Amount
filtered_df = filtered_df[
    (filtered_df["amount"] >= amt_range[0]) & 
    (filtered_df["amount"] <= amt_range[1])
]

# 6. Dates
if isinstance(selected_dates, (tuple, list)) and len(selected_dates) == 2:
    start_d, end_d = selected_dates[0], selected_dates[1]
    filtered_df = filtered_df[
        (pd.to_datetime(filtered_df["timestamp"]).dt.date >= start_d) & 
        (pd.to_datetime(filtered_df["timestamp"]).dt.date <= end_d)
    ]

# Summary metrics of filtered view
s1, s2, s3, s4 = st.columns(4)
with s1:
    render_kpi("Matching Transactions", f"{len(filtered_df):,}", f"Out of {len(df):,} total", "#38bdf8")
with s2:
    render_kpi("Filtered Volume", format_currency(filtered_df["amount"].sum()), "Filtered aggregate", "#10b981")
with s3:
    susp_match = int((filtered_df["detection_status"] == "Suspicious").sum())
    render_kpi("Suspicious Matches", f"{susp_match:,}", "Flagged high threat", "#ef4444")
with s4:
    avg_amt = filtered_df["amount"].mean() if len(filtered_df) > 0 else 0
    render_kpi("Avg Filtered Amount", format_currency(avg_amt), "Mean transaction size", "#f59e0b")

st.markdown("<br/>", unsafe_allow_html=True)

# ==========================================
# 2. FILTERABLE DATA TABLE
# ==========================================
display_cols = [
    "transaction_id", "sender_account", "receiver_account", 
    "amount", "timestamp", "transaction_type", "risk_score", "detection_status"
]
table_df = filtered_df[[c for c in display_cols if c in filtered_df.columns]].head(max_rows)

st.markdown(f"### 📋 Ledger Records ({len(table_df)} rows displayed)")
st.dataframe(
    table_df.rename(columns={
        "transaction_id": "Transaction ID",
        "sender_account": "Sender Account",
        "receiver_account": "Receiver Account",
        "amount": "Amount ($)",
        "timestamp": "Timestamp",
        "transaction_type": "Type",
        "risk_score": "Risk Score",
        "detection_status": "Detection Status"
    }),
    use_container_width=True,
    hide_index=True
)

# Export button
csv_data = filtered_df.to_csv(index=False)
st.download_button(
    label="⬇️ Export Filtered Ledger (CSV)",
    data=csv_data,
    file_name="filtered_transactions_audit.csv",
    mime="text/csv"
)

st.markdown("---")

# ==========================================
# 3. INDIVIDUAL TRANSACTION INSPECTOR
# ==========================================
st.markdown("### 🔬 Inspect Individual Transaction Details")

if len(filtered_df) > 0:
    txn_options = list(table_df["transaction_id"].unique())
    selected_txn_id = st.selectbox("Select Transaction to Inspect", txn_options)

    if selected_txn_id:
        record = df[df["transaction_id"] == selected_txn_id].iloc[0]
        sender = record["sender_account"]
        receiver = record["receiver_account"]
        amount = float(record["amount"])
        t_time = str(record["timestamp"])
        t_status = record.get("detection_status", "N/A")
        t_score = float(record.get("risk_score", 0.0))

        st.markdown(f"""
        <div style="background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 20px; margin-top: 10px;">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e293b; padding-bottom: 12px; margin-bottom: 16px;">
                <div>
                    <span style="font-size: 1.15rem; font-weight: 700; color: #38bdf8; font-family: monospace;">{selected_txn_id}</span>
                    <span style="margin-left: 12px;" class="threat-badge {'threat-high' if t_status == 'Suspicious' else 'threat-normal'}">{t_status} (Score: {t_score:.2f})</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Timestamp: <b>{t_time}</b></div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr auto 1fr; gap: 20px; align-items: center; text-align: center;">
                <div style="background: #1e293b; padding: 15px; border-radius: 10px; text-align: left;">
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Originating Sender</div>
                    <div style="font-size: 1.1rem; font-weight: bold; color: #f8fafc; font-family: monospace;">{sender}</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px;">
                        Risk Score: <b>{G.nodes[sender].get('risk_score', 0.0):.2f}</b><br/>
                        Total Volume: <b>${G.nodes[sender].get('total_volume', 0.0):,.2f}</b><br/>
                        Transfers Out: <b>{G.nodes[sender].get('out_degree', 0)}</b>
                    </div>
                </div>
                
                <div>
                    <div style="font-size: 1.3rem; font-weight: bold; color: #38bdf8;">${amount:,.2f}</div>
                    <div style="color: #64748b; font-size: 1.5rem;">&rarr;</div>
                    <div style="font-size: 0.8rem; color: #94a3b8;">{record.get('transaction_type', 'WIRE')}</div>
                </div>
                
                <div style="background: #1e293b; padding: 15px; border-radius: 10px; text-align: left;">
                    <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Beneficiary Receiver</div>
                    <div style="font-size: 1.1rem; font-weight: bold; color: #f8fafc; font-family: monospace;">{receiver}</div>
                    <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 6px;">
                        Risk Score: <b>{G.nodes[receiver].get('risk_score', 0.0):.2f}</b><br/>
                        Total Volume: <b>${G.nodes[receiver].get('total_volume', 0.0):,.2f}</b><br/>
                        Transfers In: <b>{G.nodes[receiver].get('in_degree', 0)}</b>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Entity reason codes
        if G is not None:
            s_reasons = generate_risk_reasons(sender, G)
            r_reasons = generate_risk_reasons(receiver, G)

            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown(f"#### ⚠️ Sender Threat Indicators ({sender})")
                for r in s_reasons:
                    st.markdown(f"&bull; {r}")
            with rc2:
                st.markdown(f"#### ⚠️ Receiver Threat Indicators ({receiver})")
                for r in r_reasons:
                    st.markdown(f"&bull; {r}")
else:
    st.info("No transactions match the selected filters.")
