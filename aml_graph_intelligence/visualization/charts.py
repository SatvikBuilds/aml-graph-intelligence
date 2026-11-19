"""
Financial Charts and Analytics Visualization Module
Provides sleek Plotly charts styled with dark navy and electric blue highlights.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List

THEME_COLORS = {
    "background": "#0a0f1d",
    "paper": "#0f172a",
    "text": "#f1f5f9",
    "subtext": "#94a3b8",
    "grid": "#1e293b",
    "cyan": "#06b6d4",
    "electric_blue": "#38bdf8",
    "crimson": "#ef4444",
    "emerald": "#10b981",
    "amber": "#f59e0b"
}

def create_volume_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Transaction volume over time (aggregate daily)."""
    df_temp = df.copy()
    df_temp["date"] = pd.to_datetime(df_temp["timestamp"]).dt.date
    daily = df_temp.groupby("date").agg(
        total_vol=("amount", "sum"),
        txn_count=("amount", "count")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["date"], 
        y=daily["total_vol"],
        mode="lines+markers",
        name="Daily Volume ($)",
        line=dict(color=THEME_COLORS["cyan"], width=3),
        fill="tozeroy",
        fillcolor="rgba(6, 182, 212, 0.12)",
        marker=dict(size=6, color=THEME_COLORS["electric_blue"])
    ))

    fig.update_layout(
        title="<b>Transaction Volume Dynamics</b>",
        paper_bgcolor=THEME_COLORS["background"],
        plot_bgcolor=THEME_COLORS["background"],
        font=dict(color=THEME_COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor=THEME_COLORS["grid"], title="Date"),
        yaxis=dict(showgrid=True, gridcolor=THEME_COLORS["grid"], title="Volume (USD)", tickprefix="$"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=350,
        hovermode="x unified"
    )
    return fig

def create_suspicious_vs_legit_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart for transaction classification breakdown."""
    if "is_laundering" in df.columns:
        counts = df["is_laundering"].value_counts()
        labels = ["Legitimate", "Suspicious"]
        values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    else:
        values = [len(df), 0]
        labels = ["Unlabeled", "Flagged"]

    colors = [THEME_COLORS["emerald"], THEME_COLORS["crimson"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        hole=0.62,
        marker=dict(colors=colors, line=dict(color=THEME_COLORS["paper"], width=2)),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
        textfont=dict(color=THEME_COLORS["text"], size=13)
    )])

    fig.update_layout(
        title="<b>AML Transaction Distribution</b>",
        paper_bgcolor=THEME_COLORS["background"],
        plot_bgcolor=THEME_COLORS["background"],
        font=dict(color=THEME_COLORS["text"], family="Inter, sans-serif"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=30),
        height=350
    )
    return fig

def create_risk_distribution_chart(scores: List[float]) -> go.Figure:
    """Distribution of account risk scores."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=25,
        marker=dict(
            color=THEME_COLORS["electric_blue"],
            line=dict(color=THEME_COLORS["paper"], width=1)
        ),
        opacity=0.85
    ))

    fig.update_layout(
        title="<b>Account Risk Score Spectrum</b>",
        paper_bgcolor=THEME_COLORS["background"],
        plot_bgcolor=THEME_COLORS["background"],
        font=dict(color=THEME_COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor=THEME_COLORS["grid"], title="Risk Score [0.0 - 1.0]", range=[0, 1]),
        yaxis=dict(showgrid=True, gridcolor=THEME_COLORS["grid"], title="Account Count"),
        margin=dict(l=40, r=20, t=50, b=40),
        height=350
    )
    return fig

def create_daily_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Daily suspicious vs legitimate transaction trend bar chart."""
    df_temp = df.copy()
    df_temp["date"] = pd.to_datetime(df_temp["timestamp"]).dt.date
    
    if "is_laundering" in df_temp.columns:
        daily_aml = df_temp.groupby(["date", "is_laundering"]).size().unstack(fill_value=0).reset_index()
        legit = daily_aml.get(0, pd.Series([0]*len(daily_aml)))
        susp = daily_aml.get(1, pd.Series([0]*len(daily_aml)))
    else:
        daily_aml = df_temp.groupby("date").size().reset_index(name="count")
        legit = daily_aml["count"]
        susp = pd.Series([0] * len(daily_aml))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_aml["date"], y=legit,
        name="Legitimate",
        marker_color="#1e293b"
    ))
    fig.add_trace(go.Bar(
        x=daily_aml["date"], y=susp,
        name="Suspicious (AML)",
        marker_color=THEME_COLORS["crimson"]
    ))

    fig.update_layout(
        barmode="stack",
        title="<b>Daily Threat Intelligence Trend</b>",
        paper_bgcolor=THEME_COLORS["background"],
        plot_bgcolor=THEME_COLORS["background"],
        font=dict(color=THEME_COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor=THEME_COLORS["grid"], title="Date"),
        yaxis=dict(showgrid=True, gridcolor=THEME_COLORS["grid"], title="Transactions"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=350
    )
    return fig

def create_confusion_matrix_chart(cm: List[List[int]]) -> go.Figure:
    """Confusion matrix heatmap with cell counts and percentages."""
    z = cm
    x = ["Predicted Legit", "Predicted Suspicious"]
    y = ["Actual Legit", "Actual Suspicious"]
    
    total = sum(sum(row) for row in cm)
    text_annot = [
        [f"<b>{val}</b><br>({(val/total*100):.1f}%)" for val in row]
        for row in cm
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        text=text_annot,
        texttemplate="%{text}",
        textfont=dict(size=14, color="#ffffff"),
        colorscale=[
            [0.0, "#0f172a"],
            [0.5, "#0284c7"],
            [1.0, "#ef4444"]
        ],
        showscale=False
    ))

    fig.update_layout(
        title="<b>GNN Model Confusion Matrix</b>",
        paper_bgcolor=THEME_COLORS["background"],
        plot_bgcolor=THEME_COLORS["background"],
        font=dict(color=THEME_COLORS["text"], family="Inter, sans-serif"),
        margin=dict(l=50, r=30, t=50, b=40),
        height=360
    )
    return fig

def create_training_history_chart(history: Dict[str, list]) -> go.Figure:
    """Plots training and validation loss and F1-score progression."""
    fig = go.Figure()
    
    epochs = history.get("epoch", [])
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("train_loss", []),
        name="Train Loss",
        line=dict(color=THEME_COLORS["amber"], width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_loss", []),
        name="Val Loss",
        line=dict(color=THEME_COLORS["crimson"], width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_f1", []),
        name="Val F1 Score",
        line=dict(color=THEME_COLORS["cyan"], width=3)
    ))

    fig.update_layout(
        title="<b>GNN Training & Convergence Curves</b>",
        paper_bgcolor=THEME_COLORS["background"],
        plot_bgcolor=THEME_COLORS["background"],
        font=dict(color=THEME_COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor=THEME_COLORS["grid"], title="Epoch"),
        yaxis=dict(showgrid=True, gridcolor=THEME_COLORS["grid"], title="Metric Value"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=360
    )
    return fig

def create_top_accounts_bar(top_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart for top high-risk accounts."""
    df_sorted = top_df.sort_values(by="risk_score", ascending=True)
    
    fig = go.Figure(go.Bar(
        x=df_sorted["risk_score"],
        y=df_sorted["account_id"],
        orientation="h",
        marker=dict(
            color=df_sorted["risk_score"],
            colorscale=[[0, "#f59e0b"], [1, "#ef4444"]],
            line=dict(color="#ffffff", width=1)
        ),
        text=[f"{s:.2f}" for s in df_sorted["risk_score"]],
        textposition="inside"
    ))

    fig.update_layout(
        title="<b>Highest Risk Accounts Ranking</b>",
        paper_bgcolor=THEME_COLORS["background"],
        plot_bgcolor=THEME_COLORS["background"],
        font=dict(color=THEME_COLORS["text"], family="Inter, sans-serif"),
        xaxis=dict(showgrid=True, gridcolor=THEME_COLORS["grid"], title="AML Risk Score", range=[0, 1.05]),
        yaxis=dict(showgrid=False, title=""),
        margin=dict(l=80, r=20, t=50, b=30),
        height=350
    )
    return fig
