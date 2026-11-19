"""
Financial Charts and Analytics Visualization Module
Provides premium Plotly charts styled with violet-indigo aurora palette and gold/rose accents.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, Any, List

THEME_COLORS = {
    "background":   "#0f0a1e",
    "paper":        "#1a1035",
    "surface":      "#1e1545",
    "text":         "#f1f0ff",
    "subtext":      "#9d8fc4",
    "grid":         "rgba(167, 139, 250, 0.08)",
    "violet":       "#7c3aed",
    "indigo":       "#6366f1",
    "lavender":     "#a78bfa",
    "rose":         "#f43f5e",
    "amber":        "#f59e0b",
    "emerald":      "#10b981",
    "sky":          "#38bdf8",
    "gold":         "#fbbf24",
    "pink":         "#ec4899",
}

GRADIENT_VIOLET = [[0, "#4c1d95"], [0.5, "#7c3aed"], [1, "#a78bfa"]]
FONT_FAMILY = "Sora, DM Sans, Inter, sans-serif"

def _base_layout(title: str, height: int = 350) -> dict:
    """Returns the shared dark layout config."""
    return dict(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color=THEME_COLORS["text"], family=FONT_FAMILY),
            x=0.01, xanchor="left"
        ),
        paper_bgcolor=THEME_COLORS["paper"],
        plot_bgcolor=THEME_COLORS["background"],
        font=dict(color=THEME_COLORS["text"], family=FONT_FAMILY, size=12),
        margin=dict(l=44, r=20, t=52, b=40),
        height=height,
        hovermode="x unified",
        legend=dict(
            bgcolor="rgba(26,16,53,0.7)",
            bordercolor="rgba(167,139,250,0.2)",
            borderwidth=1,
            font=dict(size=11, color=THEME_COLORS["subtext"])
        )
    )

def _base_axes(show_x_grid=True, show_y_grid=True, x_title="", y_title="") -> dict:
    return dict(
        xaxis=dict(
            showgrid=show_x_grid, gridcolor=THEME_COLORS["grid"],
            gridwidth=1, zeroline=False,
            linecolor="rgba(167,139,250,0.15)",
            tickfont=dict(color=THEME_COLORS["subtext"], size=11),
            title=dict(text=x_title, font=dict(color=THEME_COLORS["subtext"], size=11))
        ),
        yaxis=dict(
            showgrid=show_y_grid, gridcolor=THEME_COLORS["grid"],
            gridwidth=1, zeroline=False,
            linecolor="rgba(167,139,250,0.15)",
            tickfont=dict(color=THEME_COLORS["subtext"], size=11),
            title=dict(text=y_title, font=dict(color=THEME_COLORS["subtext"], size=11))
        )
    )


def create_volume_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Transaction volume over time — gradient area chart."""
    df_temp = df.copy()
    df_temp["date"] = pd.to_datetime(df_temp["timestamp"]).dt.date
    daily = df_temp.groupby("date").agg(
        total_vol=("amount", "sum"),
        txn_count=("amount", "count")
    ).reset_index()

    fig = go.Figure()

    # Volume area
    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["total_vol"],
        mode="lines",
        name="Daily Volume ($)",
        line=dict(color=THEME_COLORS["lavender"], width=2.5, shape="spline"),
        fill="tozeroy",
        fillcolor="rgba(124, 58, 237, 0.15)",
        hovertemplate="<b>%{x}</b><br>Volume: $%{y:,.0f}<extra></extra>"
    ))

    # Count bars on secondary axis
    fig.add_trace(go.Bar(
        x=daily["date"],
        y=daily["txn_count"],
        name="Txn Count",
        marker_color="rgba(99,102,241,0.3)",
        marker_line=dict(color="rgba(99,102,241,0.6)", width=1),
        yaxis="y2",
        hovertemplate="<b>%{x}</b><br>Transactions: %{y}<extra></extra>"
    ))

    layout = _base_layout("Transaction Volume Dynamics", 360)
    layout.update(_base_axes(x_title="Date", y_title="Volume (USD)"))
    layout.update(dict(
        yaxis=dict(
            showgrid=True, gridcolor=THEME_COLORS["grid"],
            zeroline=False, tickprefix="$",
            tickfont=dict(color=THEME_COLORS["subtext"], size=11),
            title=dict(text="Volume (USD)", font=dict(color=THEME_COLORS["subtext"], size=11))
        ),
        yaxis2=dict(
            overlaying="y", side="right", showgrid=False,
            tickfont=dict(color=THEME_COLORS["subtext"], size=10),
            title=dict(text="Count", font=dict(color=THEME_COLORS["subtext"], size=10))
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(26,16,53,0.7)",
            bordercolor="rgba(167,139,250,0.2)", borderwidth=1,
            font=dict(size=11)
        )
    ))
    fig.update_layout(layout)
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

    colors = [THEME_COLORS["emerald"], THEME_COLORS["rose"]]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.70,
        marker=dict(
            colors=colors,
            line=dict(color=THEME_COLORS["background"], width=4)
        ),
        textinfo="label+percent",
        hoverinfo="label+value+percent",
        textfont=dict(color=THEME_COLORS["text"], size=12, family=FONT_FAMILY),
        pull=[0, 0.05],
        rotation=90
    )])

    # Center annotation
    total = sum(values)
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:10px'>TOTAL</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=18, color=THEME_COLORS["text"], family=FONT_FAMILY),
        align="center"
    )

    layout = _base_layout("AML Classification Breakdown", 360)
    layout.update(dict(
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.18,
            xanchor="center", x=0.5,
            bgcolor="transparent",
            font=dict(size=12)
        ),
    ))
    fig.update_layout(layout)
    return fig


def create_risk_distribution_chart(scores: List[float]) -> go.Figure:
    """Distribution of account risk scores — gradient histogram."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=25,
        marker=dict(
            color=scores,
            colorscale=[
                [0.0,  "#10b981"],
                [0.35, "#6366f1"],
                [0.65, "#f59e0b"],
                [1.0,  "#f43f5e"]
            ],
            cmin=0, cmax=1,
            line=dict(color=THEME_COLORS["background"], width=1.2)
        ),
        opacity=0.9,
        hovertemplate="Risk Score: %{x:.2f}<br>Count: %{y}<extra></extra>"
    ))

    # Add vertical threshold lines
    for x, color, label in [(0.35, "#fbbf24", "Medium"), (0.70, "#f43f5e", "High")]:
        fig.add_vline(
            x=x, line_dash="dash", line_color=color,
            line_width=1.5, opacity=0.7,
            annotation_text=label,
            annotation_font=dict(color=color, size=10),
            annotation_position="top right"
        )

    layout = _base_layout("Account Risk Score Spectrum", 360)
    layout.update(_base_axes(x_title="Risk Score [0.0 – 1.0]", y_title="Account Count"))
    layout.update(dict(xaxis=dict(range=[0, 1], showgrid=True, gridcolor=THEME_COLORS["grid"],
                                   zeroline=False, tickfont=dict(color=THEME_COLORS["subtext"])),
                       hovermode="x"))
    fig.update_layout(layout)
    return fig


def create_daily_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Daily suspicious vs legitimate transaction trend."""
    df_temp = df.copy()
    df_temp["date"] = pd.to_datetime(df_temp["timestamp"]).dt.date

    if "is_laundering" in df_temp.columns:
        daily_aml = df_temp.groupby(["date", "is_laundering"]).size().unstack(fill_value=0).reset_index()
        legit = daily_aml.get(0, pd.Series([0] * len(daily_aml)))
        susp = daily_aml.get(1, pd.Series([0] * len(daily_aml)))
    else:
        daily_aml = df_temp.groupby("date").size().reset_index(name="count")
        legit = daily_aml["count"]
        susp = pd.Series([0] * len(daily_aml))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily_aml["date"], y=legit,
        name="Legitimate",
        marker_color="rgba(99,102,241,0.55)",
        marker_line=dict(width=0),
        hovertemplate="%{x}<br>Legitimate: %{y}<extra></extra>"
    ))
    fig.add_trace(go.Bar(
        x=daily_aml["date"], y=susp,
        name="Suspicious",
        marker_color=THEME_COLORS["rose"],
        marker_line=dict(width=0),
        hovertemplate="%{x}<br>Suspicious: %{y}<extra></extra>"
    ))

    layout = _base_layout("Daily Threat Intelligence Trend", 360)
    layout.update(_base_axes(x_title="Date", y_title="Transactions"))
    layout.update(dict(
        barmode="stack",
        bargap=0.15,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="transparent", font=dict(size=11)),
        hovermode="x unified"
    ))
    fig.update_layout(layout)
    return fig


def create_confusion_matrix_chart(cm: List[List[int]]) -> go.Figure:
    """Confusion matrix heatmap."""
    z = cm
    x = ["Predicted Legit", "Predicted Suspicious"]
    y = ["Actual Legit", "Actual Suspicious"]
    total = max(1, sum(sum(row) for row in cm))
    text_annot = [
        [f"<b>{val}</b><br>({(val/total*100):.1f}%)" for val in row]
        for row in cm
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=x, y=y,
        text=text_annot,
        texttemplate="%{text}",
        textfont=dict(size=14, color="#ffffff", family=FONT_FAMILY),
        colorscale=[
            [0.0, "#1a1035"],
            [0.5, "#6d28d9"],
            [1.0, "#f43f5e"]
        ],
        showscale=False
    ))

    layout = _base_layout("GNN Model Confusion Matrix", 360)
    fig.update_layout(layout)
    return fig


def create_training_history_chart(history: Dict[str, list]) -> go.Figure:
    """Training and validation loss / F1 progression."""
    fig = go.Figure()
    epochs = history.get("epoch", [])

    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("train_loss", []),
        name="Train Loss",
        line=dict(color=THEME_COLORS["amber"], width=2, dash="dot"),
        mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_loss", []),
        name="Val Loss",
        line=dict(color=THEME_COLORS["rose"], width=2.5),
        mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=history.get("val_f1", []),
        name="Val F1 Score",
        line=dict(color=THEME_COLORS["lavender"], width=3),
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(124,58,237,0.08)"
    ))

    layout = _base_layout("GNN Training & Convergence Curves", 360)
    layout.update(_base_axes(x_title="Epoch", y_title="Metric Value"))
    layout.update(dict(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="transparent", font=dict(size=11))
    ))
    fig.update_layout(layout)
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
            colorscale=[
                [0.0, "#6366f1"],
                [0.4, "#f59e0b"],
                [0.7, "#f97316"],
                [1.0, "#f43f5e"]
            ],
            cmin=0, cmax=1,
            line=dict(color="rgba(255,255,255,0.1)", width=1)
        ),
        text=[f"{s:.2f}" for s in df_sorted["risk_score"]],
        textposition="inside",
        textfont=dict(color="#ffffff", size=12, family=FONT_FAMILY),
        hovertemplate="%{y}<br>Risk Score: %{x:.3f}<extra></extra>"
    ))

    layout = _base_layout("Highest Risk Accounts Ranking", 350)
    layout.update(dict(
        xaxis=dict(
            showgrid=True, gridcolor=THEME_COLORS["grid"],
            zeroline=False, range=[0, 1.08],
            tickfont=dict(color=THEME_COLORS["subtext"], size=11),
            title=dict(text="AML Risk Score", font=dict(color=THEME_COLORS["subtext"], size=11))
        ),
        yaxis=dict(showgrid=False, tickfont=dict(color=THEME_COLORS["subtext"], size=11)),
        margin=dict(l=90, r=20, t=52, b=36),
    ))
    fig.update_layout(layout)
    return fig

