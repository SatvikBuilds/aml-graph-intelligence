"""
PAGE 6: Investigation Center
Compliance Officer Forensic Workbench, Fund Tracing, and Regulatory Dossier Reporting
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
from utils.auth import require_authentication, render_officer_sidebar, get_current_user
from database.mongo_manager import get_mongo_manager
from models.prediction import generate_risk_reasons
from graph.graph_analysis import trace_fund_flow_path, get_k_hop_neighborhood
from utils.report_generator import generate_csv_report, generate_html_dossier

st.set_page_config(
    page_title="Investigation Center | GraphAML",
    page_icon="🛡️",
    layout="wide"
)

inject_fintech_css()
initialize_session_state()
require_authentication()
render_officer_sidebar()

st.markdown("""
<div style="margin-bottom: 20px;">
    <span class="threat-badge threat-high">REGULATORY COMPLIANCE DOSSIER</span>
    <h1 style="margin: 6px 0 2px 0;">AML Case Investigation Center</h1>
    <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
        Conduct deep-dive forensic audits on flagged accounts, trace multi-hop fund origins and exits, 
        manage SAR filing workflows, and export audit-ready investigation dossiers.
    </p>
</div>
""", unsafe_allow_html=True)

G = st.session_state.get("G")
df = st.session_state.get("df")
preds_df = st.session_state.get("predictions_df")
mongo = get_mongo_manager()
current_officer = get_current_user()

# Merge MongoDB cases with session cases
mongo_cases = mongo.get_all_cases()
cases = st.session_state.get("investigation_cases", {})
cases.update(mongo_cases)

if G is None or df is None:
    st.warning("⚠️ No active transaction dataset. Please load data in Dataset Management.")
    st.stop()

# Priority account selection: Accounts with highest risk scores first
if preds_df is not None:
    sorted_accounts = preds_df.sort_values(by="risk_score", ascending=False)["account_id"].tolist()
else:
    sorted_accounts = sorted(list(G.nodes()))

# ==========================================
# 1. CASE SELECTOR & STATUS STRIP
# ==========================================
c_bar1, c_bar2, c_bar3 = st.columns([1.5, 1, 1])

with c_bar1:
    selected_account = st.selectbox(
        "Select Investigation Subject Account",
        sorted_accounts,
        index=0,
        help="Select a suspicious account to load its complete forensic file."
    )

default_officer_name = current_officer.get("full_name") or "Compliance Officer #01"
current_case = cases.get(selected_account, {
    "case_id": f"CASE-2025-{abs(hash(selected_account)) % 900 + 100:03d}",
    "account_id": selected_account,
    "status": "Under Investigation",
    "assigned_officer": default_officer_name,
    "notes": "Automated system alert triggered by multi-hop AML anomaly."
})

with c_bar2:
    status_options = ["New", "Under Investigation", "Resolved", "Flagged for SAR / STR"]
    current_status = current_case.get("status", "Under Investigation")
    s_idx = status_options.index(current_status) if current_status in status_options else 1
    new_status = st.selectbox("Case Status", status_options, index=s_idx)

with c_bar3:
    assigned_officer = st.text_input("Lead Investigator", value=current_case.get("assigned_officer", default_officer_name))

# Save status updates back to session state & MongoDB
status_changed = (new_status != current_case.get("status")) or (assigned_officer != current_case.get("assigned_officer"))
current_case["status"] = new_status
current_case["assigned_officer"] = assigned_officer
cases[selected_account] = current_case
st.session_state["investigation_cases"] = cases

if status_changed:
    mongo.save_case(current_case)
    mongo.log_audit(
        username=current_officer.get("username", "OFFICER"),
        role=current_officer.get("role", "INVESTIGATOR"),
        action="CASE_STATUS_CHANGED",
        target=selected_account,
        details=f"Status set to '{new_status}', assigned to '{assigned_officer}'."
    )

# Account metadata
node_data = G.nodes[selected_account]
score = float(node_data.get("risk_score", 0.0))
risk_level = node_data.get("risk_level", "High Risk")
reasons = generate_risk_reasons(selected_account, G)

# ==========================================
# 2. CASE DOSSIER HEADER CARD
# ==========================================
st.markdown(f"""
<div style="background: linear-gradient(135deg, #0f172a 0%, #172554 100%); border: 1px solid #1e293b; border-radius: 14px; padding: 22px; margin-bottom: 20px;">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
        <div>
            <span class="threat-badge threat-normal" style="margin-bottom: 8px;">CASE FILE: {current_case['case_id']}</span>
            <h2 style="margin: 6px 0; font-family: monospace; color: #f8fafc; font-size: 1.8rem;">{selected_account}</h2>
            <div style="color: #94a3b8; font-size: 0.9rem;">
                Assigned Officer: <b style="color: #cbd5e1;">{assigned_officer}</b> &bull; Status: <b style="color: #38bdf8;">{new_status}</b>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 0.8rem; color: #94a3b8; text-transform: uppercase;">Composite AML Threat</div>
            <div style="font-size: 2.2rem; font-weight: 700; color: {'#ef4444' if score >= 0.70 else '#f59e0b'}; font-family: 'JetBrains Mono', monospace;">
                {score:.2f}
            </div>
            <span class="threat-badge {'threat-high' if score >= 0.70 else 'threat-medium'}">{risk_level}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Metrics Strip
m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    render_kpi("Cumulative Volume", format_currency(node_data.get("total_volume", 0.0)), "Total transacted", "#38bdf8")
with m2:
    render_kpi("Funds Inflow", format_currency(node_data.get("total_received", 0.0)), f"{node_data.get('in_degree', 0)} sending nodes", "#10b981")
with m3:
    render_kpi("Funds Outflow", format_currency(node_data.get("total_sent", 0.0)), f"{node_data.get('out_degree', 0)} recipient nodes", "#ef4444")
with m4:
    render_kpi("Rapid Pass-Throughs", f"{node_data.get('rapid_pass_through', 0)}", "Dispatched < 3 hrs", "#f59e0b")
with m5:
    render_kpi("Structuring Deposits", f"{node_data.get('smurf_count', 0)}", "Deposits < $10k", "#a78bfa")

st.markdown("<br/>", unsafe_allow_html=True)

# ==========================================
# 3. COMPLIANCE REASON CODES & INVESTIGATOR NOTES
# ==========================================
sec1, sec2 = st.columns([1.2, 1])

with sec1:
    st.markdown("### ⚠️ Regulatory Flag Reasons & AI Attribution")
    for r in reasons:
        st.markdown(f"""
        <div class="alert-panel" style="margin-bottom: 8px;">
            <div style="font-size: 0.9rem; color: #f1f5f9;">&bull; {r}</div>
        </div>
        """, unsafe_allow_html=True)

with sec2:
    st.markdown("### 📝 Compliance Officer Investigation Notes")
    officer_notes = st.text_area(
        "Forensic Log & Justification",
        value=current_case.get("notes", "No officer notes recorded."),
        height=140
    )
    if st.button("💾 Save Officer Notes & Sync to MongoDB"):
        current_case["notes"] = officer_notes
        cases[selected_account] = current_case
        st.session_state["investigation_cases"] = cases
        mongo.save_case(current_case)
        mongo.log_audit(
            username=current_officer.get("username", "OFFICER"),
            role=current_officer.get("role", "INVESTIGATOR"),
            action="OFFICER_NOTES_SAVED",
            target=selected_account,
            details=f"Dossier notes updated by {current_officer.get('full_name')} for {selected_account}."
        )
        st.success("Case notes saved and synchronized with MongoDB!")

st.markdown("---")

# ==========================================
# 4. MULTI-HOP FUND TRACING ENGINE
# ==========================================
st.markdown("### 🛤️ Multi-Hop Fund Path Tracer (Source & Destination)")
st.markdown("""
<div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 12px;">
    Trace directed multi-hop monetary flow paths between the target account and any connected external entity.
</div>
""", unsafe_allow_html=True)

trace_col1, trace_col2, trace_col3 = st.columns([1.5, 1.5, 1])

with trace_col1:
    dest_candidates = [n for n in sorted_accounts if n != selected_account]
    trace_target = st.selectbox("Destination Account to Trace Flow", dest_candidates)

with trace_col3:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    run_trace_btn = st.button("🔍 Trace Directed Path", use_container_width=True)

if run_trace_btn or trace_target:
    path_result = trace_fund_flow_path(G, source=selected_account, target=trace_target)
    
    if path_result:
        st.markdown(f"""
        <div style="background: #0f172a; border: 1px solid #06b6d4; border-radius: 10px; padding: 16px; margin: 12px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; color: #38bdf8;">DIRECTED TRANSFER PATH FOUND ({path_result['hop_count']} HOPS)</span>
                <span style="font-size: 0.9rem; color: #10b981; font-weight: bold;">Path Volume: ${path_result['total_path_amount']:,.2f}</span>
            </div>
            <div style="margin-top: 10px; font-family: monospace; font-size: 0.95rem; color: #f8fafc;">
                {' &rarr; '.join(path_result['path'])}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Breakdown table
        st.dataframe(pd.DataFrame(path_result["transfers"]), use_container_width=True, hide_index=True)
    else:
        st.info(f"No direct forward path found from {selected_account} to {trace_target}. Trying reverse direction (Inflow trace)...")
        rev_path = trace_fund_flow_path(G, source=trace_target, target=selected_account)
        if rev_path:
            st.markdown(f"""
            <div style="background: #0f172a; border: 1px solid #f59e0b; border-radius: 10px; padding: 16px; margin: 12px 0;">
                <span style="font-weight: bold; color: #fbbf24;">INCOMING FLOW TRACED FROM {trace_target} ({rev_path['hop_count']} HOPS)</span>
                <div style="margin-top: 8px; font-family: monospace; font-size: 0.95rem; color: #f8fafc;">
                    {' &rarr; '.join(rev_path['path'])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(rev_path["transfers"]), use_container_width=True, hide_index=True)
        else:
            st.warning("No directed money flow path connects these two accounts in the current graph.")

st.markdown("---")

# ==========================================
# 5. ACCOUNT TRANSACTION JOURNAL
# ==========================================
st.markdown("### 📋 Account Activity Journal")

acc_txns = df[
    (df["sender_account"] == selected_account) | 
    (df["receiver_account"] == selected_account)
].sort_values(by="timestamp", ascending=False)

st.dataframe(
    acc_txns[[
        "transaction_id", "timestamp", "sender_account", "receiver_account", 
        "amount", "transaction_type", "risk_score", "detection_status"
    ]].rename(columns={
        "transaction_id": "Txn ID",
        "timestamp": "Timestamp",
        "sender_account": "Sender",
        "receiver_account": "Receiver",
        "amount": "Amount ($)",
        "transaction_type": "Type",
        "risk_score": "Risk Score",
        "detection_status": "Status"
    }),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")

# ==========================================
# 6. EXPORT INVESTIGATION DOSSIER & REPORTS
# ==========================================
st.markdown("### 📄 Export Official AML Investigation Dossier")
st.markdown("""
<div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 12px;">
    Export a standardized compliance dossier for internal audit, senior executive sign-off, 
    or submission of a Suspicious Activity Report (SAR / STR) to regulatory authorities.
</div>
""", unsafe_allow_html=True)

exp_c1, exp_c2 = st.columns(2)

with exp_c1:
    # CSV Report
    csv_report = generate_csv_report(selected_account, current_case, df)
    st.download_button(
        label="⬇️ Download Account Forensic Ledger (CSV)",
        data=csv_report,
        file_name=f"AML_Audit_{selected_account}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )

with exp_c2:
    # HTML Print-Ready Dossier
    html_dossier = generate_html_dossier(
        account_id=selected_account,
        case_info=current_case,
        node_data=node_data,
        reasons=reasons,
        txns_df=df
    )
    st.download_button(
        label="⬇️ Download Print-Ready Compliance Dossier (HTML)",
        data=html_dossier,
        file_name=f"AML_Dossier_{selected_account}_{datetime.now().strftime('%Y%m%d')}.html",
        mime="text/html",
        use_container_width=True
    )

st.markdown("---")

# ==========================================
# 7. MONGODB REGULATORY AUDIT TRAIL
# ==========================================
st.markdown("### 🍃 Live MongoDB Regulatory Audit Trail")
st.markdown("""
<div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 12px;">
    All officer logins, dossier updates, GNN model training runs, and SAR classifications 
    are persistently recorded in MongoDB collection <code>audit_logs</code>.
</div>
""", unsafe_allow_html=True)

audit_logs = mongo.get_recent_audit_logs(limit=20)
if audit_logs:
    st.dataframe(
        pd.DataFrame(audit_logs).rename(columns={
            "timestamp": "Timestamp",
            "username": "Officer ID",
            "role": "Role",
            "action": "Action Code",
            "target": "Target Entity",
            "details": "Audit Event Details"
        }),
        use_container_width=True,
        hide_index=True
    )
else:
    st.info("No audit events recorded yet.")
