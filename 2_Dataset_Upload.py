"""
PAGE 2: Dataset Upload
Interactive CSV Data Ingestion, Schema Mapping, and Quality Validation
"""

import streamlit as st
import pandas as pd
import os
import io

from utils.helpers import (
    inject_fintech_css, 
    initialize_session_state, 
    get_sample_csv_path,
    load_and_process_dataset,
    format_currency
)
from utils.auth import require_authentication, render_officer_sidebar, get_current_user
from database.mongo_manager import get_mongo_manager
from preprocessing.data_cleaning import (
    detect_column_mapping, 
    STANDARD_COLUMNS
)

st.set_page_config(
    page_title="Dataset Management | GraphAML",
    page_icon="📁",
    layout="wide"
)

inject_fintech_css()
initialize_session_state()
require_authentication()
render_officer_sidebar()

st.markdown("""
<div style="margin-bottom: 20px;">
    <span class="threat-badge threat-normal">DATA INGESTION & PIPELINE CONFIGURATION</span>
    <h1 style="margin: 6px 0 2px 0;">Dataset Management Studio</h1>
    <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
        Ingest transaction datasets (IBM AML, PaySim, Kaggle, or proprietary bank logs), 
        configure schema mappings, audit data cleanliness, and trigger graph construction.
    </p>
</div>
""", unsafe_allow_html=True)

# Two main columns: Left = Ingestion controls, Right = Data Quality & Template
left_col, right_col = st.columns([1.1, 0.9])

with left_col:
    st.markdown("### 📥 1. Upload Transaction Dataset")
    uploaded_file = st.file_uploader(
        "Upload Financial Transaction CSV", 
        type=["csv"],
        help="Upload CSV transaction records containing sender, receiver, amount, and timestamp."
    )

    # Sample dataset option
    st.markdown("---")
    st.markdown("#### Or Use Synthetic AML Benchmark Dataset")
    st.markdown("""
    <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 10px;">
        Pre-configured with realistic money laundering topologies: Smurfing fan-in clusters, 
        multi-hop layering chains, and circular round-trips.
    </div>
    """, unsafe_allow_html=True)

    use_sample_btn = st.button("🔄 Load Built-in Synthetic AML Dataset", use_container_width=True)

with right_col:
    st.markdown("### 📋 2. Template & Sample Download")
    st.markdown("""
    <div class="alert-panel">
        <b>Schema Recommendations:</b><br/>
        &bull; <b>Sender Account</b> (Mandatory): Originating account identifier<br/>
        &bull; <b>Receiver Account</b> (Mandatory): Destination account identifier<br/>
        &bull; <b>Amount</b> (Mandatory): Numeric value in currency (USD, EUR, etc.)<br/>
        &bull; <b>Timestamp</b>: ISO date string (<code>YYYY-MM-DD HH:MM:SS</code>) or time step<br/>
        &bull; <b>Transaction Type</b> (Optional): WIRE, TRANSFER, CASH_DEPOSIT, etc.<br/>
        &bull; <b>AML Label</b> (Optional): 1 for suspicious / SAR, 0 for legitimate
    </div>
    """, unsafe_allow_html=True)

    sample_path = get_sample_csv_path()
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            sample_bytes = f.read()
        st.download_button(
            label="⬇️ Download Sample AML Transactions CSV",
            data=sample_bytes,
            file_name="sample_aml_transactions.csv",
            mime="text/csv",
            use_container_width=True
        )

# Determine working dataframe
raw_df = None
file_source_name = ""

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        file_source_name = f"Uploaded File: {uploaded_file.name}"
        st.session_state["uploaded_file_info"] = {
            "name": uploaded_file.name,
            "size_kb": round(uploaded_file.size / 1024, 2)
        }
    except Exception as e:
        st.error(f"Error reading uploaded CSV: {e}")
elif use_sample_btn or st.session_state.get("raw_df") is None:
    if os.path.exists(sample_path):
        raw_df = pd.read_csv(sample_path)
        file_source_name = "Built-in Synthetic AML Dataset"
        st.session_state["uploaded_file_info"] = {
            "name": "sample_transactions.csv",
            "size_kb": round(os.path.getsize(sample_path) / 1024, 2)
        }
else:
    raw_df = st.session_state.get("raw_df")
    file_source_name = st.session_state.get("data_source", "Active Session Data")

# ==========================================
# DATASET PREVIEW & METADATA
# ==========================================
if raw_df is not None:
    st.markdown("---")
    st.markdown("### 📊 Dataset Preview & Schema Audit")
    
    file_info = st.session_state.get("uploaded_file_info", {"name": "In-Memory Dataset", "size_kb": 0.0})
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Source File", file_info["name"])
    with m_col2:
        st.metric("File Size", f"{file_info['size_kb']} KB")
    with m_col3:
        st.metric("Total Rows", f"{len(raw_df):,}")
    with m_col4:
        st.metric("Total Columns", f"{raw_df.shape[1]}")

    st.markdown("#### Raw Transaction Preview (First 5 Rows)")
    st.dataframe(raw_df.head(5), use_container_width=True)

    # Missing values and data quality
    with st.expander("🔍 Data Quality & Missing Value Inspection", expanded=False):
        null_counts = raw_df.isnull().sum()
        null_df = pd.DataFrame({
            "Column": raw_df.columns,
            "Null Count": null_counts.values,
            "Null %": (null_counts.values / len(raw_df) * 100).round(2),
            "Data Type": raw_df.dtypes.astype(str).values
        })
        st.dataframe(null_df, use_container_width=True, hide_index=True)

    # ==========================================
    # CONFIGURABLE COLUMN MAPPING INTERFACE
    # ==========================================
    st.markdown("---")
    st.markdown("### ⚙️ 3. Configure Schema Column Mapping")
    st.markdown("""
    <div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 12px;">
        Verify or adjust how the uploaded columns correspond to the core AML features. 
        Auto-detection has pre-selected matching fields.
    </div>
    """, unsafe_allow_html=True)

    detected = detect_column_mapping(list(raw_df.columns))
    df_cols = ["-- Select / None --"] + list(raw_df.columns)

    map_col1, map_col2, map_col3 = st.columns(3)
    
    with map_col1:
        # Sender
        def_sender = detected.get("sender_account") or "-- Select / None --"
        s_idx = df_cols.index(def_sender) if def_sender in df_cols else 0
        map_sender = st.selectbox("Sender Account (Required)*", df_cols, index=s_idx)

        # Receiver
        def_recv = detected.get("receiver_account") or "-- Select / None --"
        r_idx = df_cols.index(def_recv) if def_recv in df_cols else 0
        map_receiver = st.selectbox("Receiver Account (Required)*", df_cols, index=r_idx)

    with map_col2:
        # Amount
        def_amt = detected.get("amount") or "-- Select / None --"
        a_idx = df_cols.index(def_amt) if def_amt in df_cols else 0
        map_amount = st.selectbox("Transaction Amount (Required)*", df_cols, index=a_idx)

        # Timestamp
        def_ts = detected.get("timestamp") or "-- Select / None --"
        t_idx = df_cols.index(def_ts) if def_ts in df_cols else 0
        map_timestamp = st.selectbox("Timestamp / Date (Optional)", df_cols, index=t_idx)

    with map_col3:
        # Transaction Type
        def_type = detected.get("transaction_type") or "-- Select / None --"
        ty_idx = df_cols.index(def_type) if def_type in df_cols else 0
        map_type = st.selectbox("Transaction Type (Optional)", df_cols, index=ty_idx)

        # AML Label
        def_lbl = detected.get("is_laundering") or "-- Select / None --"
        l_idx = df_cols.index(def_lbl) if def_lbl in df_cols else 0
        map_label = st.selectbox("Fraud / AML Label (Optional)", df_cols, index=l_idx)

    # Validation check
    has_mandatory = (
        map_sender != "-- Select / None --" and 
        map_receiver != "-- Select / None --" and 
        map_amount != "-- Select / None --"
    )

    if not has_mandatory:
        st.warning("⚠️ Please map the 3 required fields (Sender, Receiver, Amount) to proceed.")

    st.markdown("<br/>", unsafe_allow_html=True)
    
    # Start Analysis Button
    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        start_btn = st.button(
            "🚀 Start Analysis & Construct Graph Intelligence", 
            disabled=not has_mandatory, 
            use_container_width=True
        )

    if start_btn and has_mandatory:
        with st.spinner("Processing transaction data, computing topological features, and constructing directed graph..."):
            custom_mapping = {
                "sender_account": None if map_sender == "-- Select / None --" else map_sender,
                "receiver_account": None if map_receiver == "-- Select / None --" else map_receiver,
                "amount": None if map_amount == "-- Select / None --" else map_amount,
                "timestamp": None if map_timestamp == "-- Select / None --" else map_timestamp,
                "transaction_type": None if map_type == "-- Select / None --" else map_type,
                "is_laundering": None if map_label == "-- Select / None --" else map_label
            }

            try:
                load_and_process_dataset(raw_df, mapping=custom_mapping, source_name=file_source_name)
                
                # Log audit event in MongoDB
                current_u = get_current_user()
                get_mongo_manager().log_audit(
                    username=current_u.get("username", "OFFICER"),
                    role=current_u.get("role", "INVESTIGATOR"),
                    action="DATASET_ANALYZED",
                    target=file_source_name,
                    details=f"Processed {len(raw_df):,} transactions. Constructed graph."
                )

                st.success("✅ Analysis completed successfully! Graph built and risk signatures evaluated.")
                
                # Jump link buttons
                st.markdown("""
                <div style="background: #0f172a; border: 1px solid #10b981; border-radius: 10px; padding: 16px; margin-top: 15px;">
                    <h4 style="color: #34d399; margin: 0 0 8px 0;">🎉 Intelligence Pipeline Ready</h4>
                    <p style="font-size: 0.9rem; color: #cbd5e1; margin: 0 0 12px 0;">
                        The graph network has been populated with accounts and directed financial transfers. 
                        Choose your next investigation workspace:
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                j1, j2, j3 = st.columns(3)
                with j1:
                    st.page_link("pages/1_Dashboard.py", label="View Executive Dashboard &rarr;", icon="📊")
                with j2:
                    st.page_link("pages/4_Graph_Visualization.py", label="Open Interactive Graph &rarr;", icon="🕸️")
                with j3:
                    st.page_link("pages/5_AI_Detection.py", label="Train GNN Models &rarr;", icon="🧠")
            except Exception as e:
                st.error(f"Error during graph intelligence pipeline: {e}")
