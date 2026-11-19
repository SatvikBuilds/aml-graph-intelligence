"""
Interactive Graph Network Visualization Module
Generates physics-based PyVis interactive graphs and Plotly network diagrams
with dark financial intelligence styling and risk-level color schemes.
"""

import os
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, Optional, List
from pyvis.network import Network

def render_pyvis_network(
    G: nx.DiGraph,
    height: str = "720px",
    width: str = "100%",
    highlight_nodes: Optional[List[str]] = None,
    highlight_edges: Optional[List[tuple]] = None,
    physics_enabled: bool = True
) -> str:
    """
    Renders an interactive PyVis network with custom dark navy fintech theme.
    Returns raw HTML string ready for Streamlit components.v1.html.
    """
    # Create pyvis network
    net = Network(
        height=height, 
        width=width, 
        directed=True, 
        bgcolor="#0a0f1d", 
        font_color="#f1f5f9"
    )

    highlight_nodes_set = set(highlight_nodes or [])
    highlight_edges_set = set(highlight_edges or [])

    # Add Nodes
    for node, data in G.nodes(data=True):
        score = data.get("risk_score", 0.0)
        risk_level = data.get("risk_level", "Normal")
        color = data.get("color", "#38bdf8")
        size = data.get("size", 16)
        label = str(node)

        # Highlight focal / searched nodes
        if node in highlight_nodes_set:
            color = "#f43f5e"  # Vivid crimson
            size = size + 10
            border_width = 3
        else:
            border_width = 1

        title_html = (
            f"<div style='font-family:Inter,sans-serif; padding:6px; font-size:12px; color:#f8fafc;'>"
            f"<strong style='color:#38bdf8; font-size:13px;'>Account: {node}</strong><br/>"
            f"Risk Level: <b style='color:{color};'>{risk_level} ({score:.2f})</b><br/>"
            f"Total Volume: <b>${data.get('total_volume', 0.0):,.2f}</b><br/>"
            f"Transfers In: <b>{data.get('in_degree', 0)}</b> | Out: <b>{data.get('out_degree', 0)}</b><br/>"
            f"Rapid Velocity: <b>{data.get('rapid_pass_through', 0)} times</b>"
            f"</div>"
        )

        net.add_node(
            node,
            label=label,
            title=title_html,
            color={
                "background": color,
                "border": "#ffffff" if node in highlight_nodes_set else color,
                "highlight": {"background": "#38bdf8", "border": "#ffffff"}
            },
            size=size,
            borderWidth=border_width,
            shape="dot",
            font={"color": "#e2e8f0", "size": 11, "face": "Inter, sans-serif"}
        )

    # Add Edges
    for u, v, data in G.edges(data=True):
        amt = data.get("amount", 0.0)
        is_suspicious = data.get("is_suspicious", 0)
        width = data.get("width", 1.5)

        is_highlighted_edge = (u, v) in highlight_edges_set or (u in highlight_nodes_set and v in highlight_nodes_set)
        
        if is_highlighted_edge:
            edge_color = "#f43f5e"
            width = width + 2
        elif is_suspicious:
            edge_color = "#ef4444"
        else:
            edge_color = "#334155"

        edge_title = (
            f"<div style='font-family:Inter,sans-serif; padding:5px; font-size:11px; color:#f8fafc;'>"
            f"Transfer: <b>{u} &rarr; {v}</b><br/>"
            f"Total Amount: <b style='color:#38bdf8;'>${amt:,.2f}</b><br/>"
            f"Txn Count: <b>{data.get('txn_count', 1)}</b><br/>"
            f"Latest: <b>{data.get('latest_timestamp', 'N/A')}</b><br/>"
            f"Status: <b style='color:{'#ef4444' if is_suspicious else '#10b981'};'>{'SUSPICIOUS' if is_suspicious else 'Normal'}</b>"
            f"</div>"
        )

        net.add_edge(
            u,
            v,
            title=edge_title,
            color=edge_color,
            width=width,
            arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
            smooth={"type": "curvedCW", "roundness": 0.15}
        )

    # Configure physics & interactions
    physics_options = f"""
    var options = {{
      "nodes": {{
        "shadow": true
      }},
      "edges": {{
        "smooth": {{ "type": "continuous" }}
      }},
      "physics": {{
        "enabled": {'true' if physics_enabled else 'false'},
        "barnesHut": {{
          "gravitationalConstant": -5000,
          "centralGravity": 0.35,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09
        }},
        "minVelocity": 0.75
      }},
      "interaction": {{
        "hover": true,
        "navigationButtons": true,
        "keyboard": true,
        "zoomView": true
      }}
    }};
    network.setOptions(options);
    """
    
    html = net.generate_html()
    
    # Inject custom styling and JS options into generated HTML
    custom_css = """
    <style>
      body { margin: 0; background-color: #0a0f1d; }
      #mynetwork { border: 1px solid #1e293b; border-radius: 12px; }
      div.vis-tooltip {
        background-color: #0f172a !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.4) !important;
      }
    </style>
    """
    html = html.replace("<style>", custom_css + "<style>")
    html = html.replace("network.setOptions(options);", physics_options)

    return html

def render_plotly_network(
    G: nx.DiGraph,
    highlight_nodes: Optional[List[str]] = None
) -> go.Figure:
    """
    Alternative Plotly 2D WebGL-accelerated interactive network graph.
    """
    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#0a0f1d",
            plot_bgcolor="#0a0f1d",
            font_color="#94a3b8",
            annotations=[dict(text="No Graph Data Available", showarrow=False, font=dict(size=16))]
        )
        return fig

    # Layout coordinates
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    # Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="#334155"),
        hoverinfo="none",
        mode="lines"
    )

    # Node traces categorized by risk
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_texts = []
    highlight_set = set(highlight_nodes or [])

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        data = G.nodes[node]
        score = data.get("risk_score", 0.0)
        risk_level = data.get("risk_level", "Normal")
        color = "#f43f5e" if node in highlight_set else data.get("color", "#38bdf8")
        
        node_colors.append(color)
        node_sizes.append(18 if node in highlight_set else 12)
        
        node_texts.append(
            f"Account: <b>{node}</b><br>"
            f"Risk: <b>{risk_level} ({score:.2f})</b><br>"
            f"Volume: ${data.get('total_volume', 0.0):,.2f}<br>"
            f"Degree: In={data.get('in_degree', 0)}, Out={data.get('out_degree', 0)}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[n if n in highlight_set else "" for n in G.nodes()],
        textposition="top center",
        hovertext=node_texts,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1.5, color="#f8fafc")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=10, l=10, r=10, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor="#0a0f1d",
        plot_bgcolor="#0a0f1d",
        height=680
    )
    return fig
