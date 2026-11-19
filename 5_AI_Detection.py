"""
PAGE 5: AI Detection Results & GNN Model Studio
Trains and Evaluates Graph Neural Networks (GCN & GAT) via PyTorch Geometric
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
from utils.auth import require_authentication, render_officer_sidebar, get_current_user
from database.mongo_manager import get_mongo_manager
from visualization.charts import (
    create_confusion_matrix_chart, 
    create_training_history_chart,
    create_top_accounts_bar
)
from models.train import prepare_pyg_data, train_gnn_model
from models.prediction import (
    predict_with_gnn, 
    predict_baseline_heuristic, 
    score_transactions, 
    generate_risk_reasons
)
from graph.graph_analysis import (
    detect_circular_flows, 
    detect_layering_chains,
    detect_smurfing_patterns
)

st.set_page_config(
    page_title="AI Detection Studio | GraphAML",
    page_icon="🧠",
    layout="wide"
)

inject_fintech_css()
initialize_session_state()
require_authentication()
render_officer_sidebar()

st.markdown("""
<div style="margin-bottom: 20px;">
    <span class="threat-badge threat-normal">DEEP GRAPH LEARNING BENCHMARK</span>
    <h1 style="margin: 6px 0 2px 0;">AI Threat Detection & GNN Studio</h1>
    <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
        Train and evaluate genuine Graph Convolutional (GCN) and Graph Attention (GAT) models 
        to classify illicit nodes, identify suspicious transaction chains, and compute network risk embeddings.
    </p>
</div>
""", unsafe_allow_html=True)

G = st.session_state.get("G")
df = st.session_state.get("df")
node_df = st.session_state.get("node_df")
meta = st.session_state.get("meta")

if G is None or df is None:
    st.warning("⚠️ No active dataset. Please load or upload dataset in Dataset Management.")
    st.stop()

# ==========================================
# 1. MODEL SELECTION & TRAINING CONTROLS
# ==========================================
with st.expander("⚙️ GNN Architecture & Training Configuration", expanded=True):
    train_c1, train_c2, train_c3, train_c4 = st.columns(4)

    with train_c1:
        model_choice = st.selectbox(
            "Select GNN Architecture",
            ["GCN (Graph Convolutional Network)", "GAT (Graph Attention Network)"]
        )
        selected_model_type = "GAT" if "GAT" in model_choice else "GCN"

    with train_c2:
        epochs = st.slider("Training Epochs", min_value=15, max_value=120, value=50, step=5)

    with train_c3:
        learning_rate = st.select_slider("Learning Rate", options=[0.001, 0.005, 0.01, 0.02, 0.05], value=0.01)

    with train_c4:
        hidden_dim = st.selectbox("Hidden Dimension Channels", [32, 64, 128], index=1)

    st.markdown("""
    <div style="font-size: 0.85rem; color: #94a3b8; margin: 6px 0 14px 0;">
        <b>Training Protocol:</b> 60% Train &bull; 20% Validation &bull; 20% Holdout Test. 
        Stratified sampling prevents data leakage. Class-weighted Cross-Entropy loss counters label imbalance.
    </div>
    """, unsafe_allow_html=True)

    t_btn_col1, t_btn_col2, t_btn_col3 = st.columns([1, 2, 1])
    with t_btn_col2:
        train_button = st.button("🚀 Train Selected GNN Model", use_container_width=True)

# Handle training
if train_button:
    with st.spinner(f"Preparing PyG Graph Tensors and Training {selected_model_type} Model..."):
        try:
            feature_cols = meta["feature_cols"] if meta else [
                "in_degree", "out_degree", "total_degree", "total_received", "total_sent",
                "avg_received", "avg_sent", "in_out_degree_ratio", "amount_in_out_ratio",
                "rapid_pass_through_count", "smurf_in_count", "pagerank", 
                "degree_centrality", "in_centrality", "out_centrality"
            ]

            # Prepare data
            data, scaler, acc_to_idx = prepare_pyg_data(G, node_df, feature_cols)

            # Training progress bar
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def progress_callback(epoch, total_epochs, metrics):
                frac = epoch / total_epochs
                progress_bar.progress(frac)
                status_text.text(f"Epoch {epoch}/{total_epochs} - Train Loss: {metrics['train_loss']:.4f} | Val F1: {metrics['val_f1']:.4f}")

            model, history, eval_metrics = train_gnn_model(
                data=data,
                model_type=selected_model_type,
                epochs=epochs,
                lr=learning_rate,
                hidden_dim=hidden_dim,
                callback_fn=progress_callback
            )

            progress_bar.progress(1.0)
            status_text.empty()

            # Generate predictions with trained GNN
            predictions = predict_with_gnn(model, data, model_name=selected_model_type)

            # Score transactions
            account_score_dict = dict(zip(predictions["account_id"], predictions["risk_score"]))
            scored_df = score_transactions(df, account_score_dict)

            # Save in session state
            st.session_state["trained_model"] = model
            st.session_state["trained_model_name"] = selected_model_type
            st.session_state["training_history"] = history
            st.session_state["model_metrics"] = eval_metrics
            st.session_state["predictions_df"] = predictions
            st.session_state["df"] = scored_df

            # Log to MongoDB
            cur_u = get_current_user()
            get_mongo_manager().log_audit(
                username=cur_u.get("username", "OFFICER"),
                role=cur_u.get("role", "INVESTIGATOR"),
                action="GNN_MODEL_TRAINED",
                target=selected_model_type,
                details=f"Trained {selected_model_type} ({epochs} epochs). Acc: {eval_metrics['accuracy']:.2%}, F1: {eval_metrics['f1_score']:.2%}"
            )

            st.success(f"🎉 {selected_model_type} Model successfully trained and evaluated on test holdout!")
        except Exception as e:
            st.error(f"Training failed: {e}")

# ==========================================
# 2. ACTIVE MODEL STATUS STRIP
# ==========================================
active_model_name = st.session_state.get("trained_model_name")
model_metrics = st.session_state.get("model_metrics")
history = st.session_state.get("training_history")
preds_df = st.session_state.get("predictions_df")

if active_model_name and model_metrics:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #064e3b, #0f172a); border: 1px solid #10b981; border-radius: 12px; padding: 16px 22px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="threat-badge threat-low">ACTIVE PRODUCTION MODEL</span>
                <h3 style="margin: 4px 0 0 0; color: #34d399;">{active_model_name} (PyTorch Geometric)</h3>
                <span style="font-size: 0.85rem; color: #cbd5e1;">Evaluated on {model_metrics['test_samples']} holdout test samples ({model_metrics['fraud_in_test']} positive AML accounts)</span>
            </div>
            <div style="display: flex; gap: 20px; font-family: 'JetBrains Mono', monospace;">
                <div>Accuracy: <b style="color: #38bdf8;">{model_metrics['accuracy']:.2%}</b></div>
                <div>Precision: <b style="color: #34d399;">{model_metrics['precision']:.2%}</b></div>
                <div>Recall: <b style="color: #fbbf24;">{model_metrics['recall']:.2%}</b></div>
                <div>F1-Score: <b style="color: #f87171;">{model_metrics['f1_score']:.2%}</b></div>
                <div>ROC-AUC: <b style="color: #a78bfa;">{model_metrics['roc_auc']:.2f}</b></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        render_kpi("Test Accuracy", f"{model_metrics['accuracy']:.2%}", "Out-of-sample accuracy", "#38bdf8")
    with m2:
        render_kpi("Precision", f"{model_metrics['precision']:.2%}", "True positive purity", "#34d399")
    with m3:
        render_kpi("Recall", f"{model_metrics['recall']:.2%}", "Laundering detection rate", "#fbbf24")
    with m4:
        render_kpi("F1-Score", f"{model_metrics['f1_score']:.2%}", "Harmonic mean", "#ef4444")
    with m5:
        render_kpi("ROC-AUC", f"{model_metrics['roc_auc']:.2f}", "Area under curve", "#a78bfa")

    # Training Curves & Confusion Matrix
    chart_c1, chart_c2 = st.columns(2)
    with chart_c1:
        st.plotly_chart(create_training_history_chart(history), use_container_width=True)
    with chart_c2:
        st.plotly_chart(create_confusion_matrix_chart(model_metrics["confusion_matrix"]), use_container_width=True)

else:
    # BASELINE HEURISTIC MODE
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #38bdf8; border-radius: 12px; padding: 16px 22px; margin-bottom: 20px;">
        <span class="threat-badge threat-normal">DEMONSTRATION BASELINE ACTIVE</span>
        <h3 style="margin: 4px 0 0 0; color: #38bdf8;">Baseline Graph Topological & Rule Heuristic</h3>
        <p style="font-size: 0.88rem; color: #94a3b8; margin: 4px 0 0 0;">
            <b>Notice:</b> A deep GNN has not been trained yet in this session. The system is operating in 
            <b>Transparent Baseline Mode</b>, computing multi-hop heuristics (structuring fan-in, layering velocity, and degree ratios).
            Click <b>"Train Selected GNN Model"</b> above to train genuine GCN or GAT architectures.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# 3. DETECTED SUSPICIOUS TRANSACTION CHAINS & MULTI-HOP TOPOLOGIES
# ==========================================
st.markdown("### 🕸️ Multi-Hop Laundering Chains & Network Topologies")

tab_chains, tab_cycles, tab_smurfs = st.tabs([
    "Multi-Hop Layering Chains", 
    "Circular Round-Tripping Loops", 
    "Smurfing Structuring Aggregators"
])

with tab_chains:
    chains = detect_layering_chains(G, min_hops=3, limit=8)
    if chains:
        for idx, ch in enumerate(chains):
            st.markdown(f"""
            <div class="alert-panel" style="margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #f87171;">Layering Chain #{idx + 1} ({ch['hop_count']} Hops)</span>
                    <span style="font-size: 0.85rem; color: #38bdf8; font-weight: bold;">Total Flow: ${ch['total_flow']:,.2f}</span>
                </div>
                <div style="margin-top: 8px; font-family: monospace; font-size: 0.85rem; color: #cbd5e1;">
                    {ch['chain_str']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No multi-hop layering chains detected.")

with tab_cycles:
    cycles = detect_circular_flows(G, max_cycle_length=5, limit=8)
    if cycles:
        for idx, cy in enumerate(cycles):
            st.markdown(f"""
            <div class="alert-panel" style="margin-bottom: 12px; border-left-color: #f59e0b;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #fbbf24;">Round-Tripping Cycle #{idx + 1} ({cy['length']} Hops Closed Loop)</span>
                    <span style="font-size: 0.85rem; color: #38bdf8; font-weight: bold;">Cycle Volume: ${cy['total_volume']:,.2f}</span>
                </div>
                <div style="margin-top: 8px; font-family: monospace; font-size: 0.85rem; color: #cbd5e1;">
                    {cy['cycle_str']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No closed round-tripping cycles found.")

with tab_smurfs:
    smurfs = detect_smurfing_patterns(G, min_in_degree=3)
    if smurfs:
        for idx, sm in enumerate(smurfs[:8]):
            st.markdown(f"""
            <div class="alert-panel" style="margin-bottom: 12px; border-left-color: #ef4444;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; color: #f87171;">Aggregator: <code style="color: #f1f5f9;">{sm['aggregator_account']}</code></span>
                    <span style="font-size: 0.85rem; color: #38bdf8; font-weight: bold;">Aggregated: ${sm['total_aggregated']:,.2f}</span>
                </div>
                <div style="margin-top: 6px; font-size: 0.85rem; color: #cbd5e1;">
                    Aggregating funds from <b>{sm['smurf_count']} smurf senders</b> with average deposit of <b>${sm['avg_deposit']:,.2f}</b> (below $10,000 threshold).
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No smurfing structuring aggregators detected.")

st.markdown("---")

# ==========================================
# 4. HIGH-RISK ACCOUNT PREDICTION RANKINGS
# ==========================================
st.markdown("### 🏆 Comprehensive Account Threat Ranking")

if preds_df is not None:
    rank_col1, rank_col2 = st.columns([1, 1])

    with rank_col1:
        top_10 = preds_df.sort_values(by="risk_score", ascending=False).head(10)
        st.plotly_chart(create_top_accounts_bar(top_10), use_container_width=True)

    with rank_col2:
        st.markdown("#### High-Threat Account Table")
        st.dataframe(
            preds_df.sort_values(by="risk_score", ascending=False).head(15).rename(columns={
                "account_id": "Account ID",
                "risk_score": "Risk Score",
                "risk_level": "Category",
                "predicted_class": "Class (1=AML)",
                "detection_mode": "Model Engine"
            }),
            use_container_width=True,
            hide_index=True
        )
