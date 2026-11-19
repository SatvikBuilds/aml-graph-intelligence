"""
Interactive Graph Network Visualization Module
Generates physics-based PyVis interactive graphs and Plotly network diagrams
with premium violet-indigo aurora theme and rose/amber risk-level color schemes.
"""

import os
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, Optional, List
from pyvis.network import Network

GRAPH_BG = "#0f0a1e"

def render_pyvis_network(
    G: nx.DiGraph,
    height: str = "720px",
    width: str = "100%",
    highlight_nodes: Optional[List[str]] = None,
    highlight_edges: Optional[List[tuple]] = None,
    physics_enabled: bool = True
) -> str:
    """
    Renders an interactive PyVis network with premium violet-indigo aurora theme.
    Returns raw HTML string ready for Streamlit components.v1.html.
    """
    net = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor=GRAPH_BG,
        font_color="#f1f0ff"
    )

    highlight_nodes_set = set(highlight_nodes or [])
    highlight_edges_set = set(highlight_edges or [])

    # Add Nodes
    for node, data in G.nodes(data=True):
        score = data.get("risk_score", 0.0)
        risk_level = data.get("risk_level", "Normal")
        color = data.get("color", "#7c3aed")
        size = data.get("size", 16)
        label = str(node)

        if node in highlight_nodes_set:
            color = "#ec4899"   # Hot pink highlight
            size = size + 10
            border_width = 3
        else:
            border_width = 1

        title_html = (
            f"<div style='font-family:DM Sans,Inter,sans-serif; padding:10px; font-size:12px; color:#f1f0ff; line-height:1.6; background:#1a1035; border-radius:10px;'>"
            f"<strong style='color:#a78bfa; font-size:13px; font-family:DM Mono,monospace;'>Account: {node}</strong><br/>"
            f"Risk Level: <b style='color:{color};'>{risk_level} ({score:.2f})</b><br/>"
            f"Total Volume: <b style='color:#818cf8;'>${data.get('total_volume', 0.0):,.2f}</b><br/>"
            f"Transfers In: <b>{data.get('in_degree', 0)}</b> | Out: <b>{data.get('out_degree', 0)}</b><br/>"
            f"Rapid Velocity: <b>{data.get('rapid_pass_through', 0)} events</b>"
            f"</div>"
        )

        net.add_node(
            node,
            label=label,
            title=title_html,
            color={
                "background": color,
                "border": "#ffffff" if node in highlight_nodes_set else color,
                "highlight": {"background": "#ec4899", "border": "#ffffff"}
            },
            size=size,
            borderWidth=border_width,
            shape="dot",
            font={"color": "#e9d5ff", "size": 11, "face": "DM Mono, monospace"}
        )

    # Add Edges
    for u, v, data in G.edges(data=True):
        amt = data.get("amount", 0.0)
        is_suspicious = data.get("is_suspicious", 0)
        width = data.get("width", 1.5)

        is_highlighted_edge = (u, v) in highlight_edges_set or (u in highlight_nodes_set and v in highlight_nodes_set)
        
        if is_highlighted_edge:
            edge_color = "#ec4899"
            width = width + 2.5
        elif is_suspicious:
            edge_color = "#f43f5e"
        else:
            edge_color = "rgba(109, 40, 217, 0.35)"

        edge_title = (
            f"<div style='font-family:DM Sans,Inter,sans-serif; padding:8px; font-size:11px; color:#f1f0ff; background:#1a1035; border-radius:8px;'>"
            f"Transfer: <b>{u} &rarr; {v}</b><br/>"
            f"Total Amount: <b style='color:#a78bfa;'>${amt:,.2f}</b><br/>"
            f"Txn Count: <b>{data.get('txn_count', 1)}</b><br/>"
            f"Latest: <b>{data.get('latest_timestamp', 'N/A')}</b><br/>"
            f"Status: <b style='color:{'#f43f5e' if is_suspicious else '#10b981'};'>{'SUSPICIOUS' if is_suspicious else 'Normal'}</b>"
            f"</div>"
        )

        net.add_edge(
            u,
            v,
            title=edge_title,
            color=edge_color,
            width=width,
            arrows={"to": {"enabled": True, "scaleFactor": 0.65}},
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
          "gravitationalConstant": -5200,
          "centralGravity": 0.32,
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
    
    # Inject premium styling
    custom_css = """
    <style>
      body { margin: 0; background-color: #0f0a1e; }
      #mynetwork {
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 16px;
        box-shadow: 0 0 60px rgba(124, 58, 237, 0.08), inset 0 0 40px rgba(0,0,0,0.5);
      }
      div.vis-tooltip {
        background-color: #1a1035 !important;
        border: 1px solid rgba(167,139,250,0.4) !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.25) !important;
        color: #f1f0ff !important;
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
    Alternative Plotly 2D interactive network graph with premium violet theme.
    """
    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=GRAPH_BG,
            plot_bgcolor=GRAPH_BG,
            font_color="#9d8fc4",
            annotations=[dict(text="No Graph Data Available", showarrow=False, font=dict(size=16))]
        )
        return fig

    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="rgba(109, 40, 217, 0.4)"),
        hoverinfo="none",
        mode="lines"
    )

    node_x, node_y, node_colors, node_sizes, node_texts = [], [], [], [], []
    highlight_set = set(highlight_nodes or [])

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        data = G.nodes[node]
        score = data.get("risk_score", 0.0)
        risk_level = data.get("risk_level", "Normal")
        color = "#ec4899" if node in highlight_set else data.get("color", "#7c3aed")
        
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
            line=dict(width=1.5, color="#ffffff")
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=10, l=10, r=10, t=10),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor=GRAPH_BG,
        plot_bgcolor=GRAPH_BG,
        height=680
    )
    return fig
