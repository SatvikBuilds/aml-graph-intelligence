"""
Investigation Report Generator Module
Creates downloadable CSV and print-ready HTML AML dossier reports
for regulatory compliance and Suspicious Activity Reports (SAR).
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

def generate_csv_report(account_id: str, case_info: Dict[str, Any], txns_df: pd.DataFrame) -> str:
    """Generates CSV report string for compliance auditing."""
    acc_txns = txns_df[
        (txns_df["sender_account"] == account_id) | 
        (txns_df["receiver_account"] == account_id)
    ]
    return acc_txns.to_csv(index=False)

def generate_html_dossier(
    account_id: str,
    case_info: Dict[str, Any],
    node_data: Dict[str, Any],
    reasons: List[str],
    txns_df: pd.DataFrame
) -> str:
    """
    Builds a professional, print-ready HTML AML Investigation Dossier.
    """
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    case_id = case_info.get("case_id", "CASE-2025-001")
    officer = case_info.get("assigned_officer", "Compliance Officer")
    status = case_info.get("status", "Under Investigation")
    score = node_data.get("risk_score", 0.0)
    risk_level = node_data.get("risk_level", "High Risk")
    notes = case_info.get("notes", "No officer notes recorded.")

    # Transactions for this account
    acc_txns = txns_df[
        (txns_df["sender_account"] == account_id) | 
        (txns_df["receiver_account"] == account_id)
    ].sort_values(by="timestamp", ascending=False).head(20)

    # Build rows
    txn_rows = []
    for _, r in acc_txns.iterrows():
        is_sender = (r["sender_account"] == account_id)
        counterparty = r["receiver_account"] if is_sender else r["sender_account"]
        flow_type = "OUTGOING" if is_sender else "INCOMING"
        color = "#ef4444" if is_sender else "#10b981"
        
        txn_rows.append(f"""
        <tr>
            <td style="padding: 8px; border-bottom: 1px solid #e2e8f0; font-family: monospace;">{r.get('transaction_id', 'N/A')}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">{r.get('timestamp', '')}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e2e8f0; font-weight: bold; color: {color};">{flow_type}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e2e8f0; font-family: monospace;">{counterparty}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e2e8f0; text-align: right; font-weight: bold;">${float(r.get('amount', 0)):,.2f}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">{r.get('transaction_type', 'TRANSFER')}</td>
            <td style="padding: 8px; border-bottom: 1px solid #e2e8f0;">{r.get('detection_status', 'N/A')}</td>
        </tr>
        """)

    reasons_html = "".join([f"<li style='margin-bottom: 6px;'>{r}</li>" for r in reasons])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8"/>
        <title>AML Investigation Dossier - {account_id}</title>
        <style>
            body {{
                font-family: 'Helvetica Neue', Arial, sans-serif;
                margin: 40px;
                color: #1e293b;
                line-height: 1.5;
            }}
            .header {{
                border-bottom: 3px solid #0284c7;
                padding-bottom: 15px;
                margin-bottom: 25px;
            }}
            .header h1 {{
                margin: 0;
                color: #0f172a;
                font-size: 24px;
            }}
            .meta-grid {{
                display: grid;
                grid-template-columns: repeat(4, 1fr);
                gap: 15px;
                background-color: #f8fafc;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 25px;
                border: 1px solid #e2e8f0;
            }}
            .meta-item label {{
                display: block;
                font-size: 11px;
                text-transform: uppercase;
                color: #64748b;
                font-weight: bold;
            }}
            .meta-item value {{
                font-size: 15px;
                font-weight: bold;
                color: #0f172a;
            }}
            .section {{
                margin-bottom: 25px;
            }}
            .section h2 {{
                font-size: 16px;
                border-bottom: 1px solid #cbd5e1;
                padding-bottom: 6px;
                color: #0f172a;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                font-size: 12px;
                margin-top: 10px;
            }}
            th {{
                background-color: #f1f5f9;
                text-align: left;
                padding: 8px;
                font-size: 11px;
                text-transform: uppercase;
                border-bottom: 2px solid #cbd5e1;
            }}
            .badge-risk {{
                background-color: #fee2e2;
                color: #b91c1c;
                padding: 4px 10px;
                border-radius: 4px;
                font-weight: bold;
            }}
            .footer {{
                margin-top: 40px;
                padding-top: 15px;
                border-top: 1px solid #e2e8f0;
                font-size: 11px;
                color: #94a3b8;
                display: flex;
                justify-content: space-between;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>FINANCIAL INTELLIGENCE UNIT &bull; AML INVESTIGATION DOSSIER</h1>
            <p style="margin: 4px 0 0 0; color: #64748b; font-size: 13px;">Dynamic Graph Intelligence for Multi-Hop Anti-Money Laundering Detection</p>
        </div>

        <div class="meta-grid">
            <div class="meta-item">
                <label>Case Reference</label>
                <value>{case_id}</value>
            </div>
            <div class="meta-item">
                <label>Subject Account</label>
                <value style="font-family: monospace;">{account_id}</value>
            </div>
            <div class="meta-item">
                <label>Investigator</label>
                <value>{officer}</value>
            </div>
            <div class="meta-item">
                <label>Case Status</label>
                <value>{status}</value>
            </div>
            <div class="meta-item">
                <label>Threat Score</label>
                <value><span class="badge-risk">{risk_level} ({score:.2f})</span></value>
            </div>
            <div class="meta-item">
                <label>Total Volume</label>
                <value>${float(node_data.get('total_volume', 0)):,.2f}</value>
            </div>
            <div class="meta-item">
                <label>Inflow / Outflow</label>
                <value>${float(node_data.get('total_received', 0)):,.2f} / ${float(node_data.get('total_sent', 0)):,.2f}</value>
            </div>
            <div class="meta-item">
                <label>Generated On</label>
                <value>{now_str}</value>
            </div>
        </div>

        <div class="section">
            <h2>1. AI & Graph Intelligence Reason Codes</h2>
            <ul style="color: #334155; font-size: 13px;">
                {reasons_html}
            </ul>
        </div>

        <div class="section">
            <h2>2. Investigator Notes & Action Plan</h2>
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; font-size: 13px; color: #334155;">
                {notes}
            </div>
        </div>

        <div class="section">
            <h2>3. Transaction Journal (Recent Activity)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Txn Ref</th>
                        <th>Timestamp</th>
                        <th>Direction</th>
                        <th>Counterparty</th>
                        <th style="text-align: right;">Amount</th>
                        <th>Type</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(txn_rows)}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <span>CONFIDENTIAL &bull; FOR INTERNAL COMPLIANCE & LAW ENFORCEMENT REPORTING ONLY</span>
            <span>Report Hash: AGY-AML-{abs(hash(account_id + str(now_str))) % 100000000}</span>
        </div>
    </body>
    </html>
    """
    return html
