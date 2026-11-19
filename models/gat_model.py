"""
Graph Attention Network (GAT) Architecture
Uses PyTorch Geometric GATConv layers with multi-head attention
to learn relational attention weights across transaction edges.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATClassifier(nn.Module):
    """
    Multi-Head Graph Attention Network for AML Risk Classification
    """
    def __init__(
        self, 
        in_channels: int, 
        hidden_dim: int = 32, 
        heads: int = 4, 
        num_classes: int = 2, 
        dropout: float = 0.3
    ):
        super(GATClassifier, self).__init__()
        # Layer 1: Multi-head attention (hidden_dim * heads)
        self.gat1 = GATConv(in_channels, hidden_dim, heads=heads, dropout=dropout)
        self.bn1 = nn.BatchNorm1d(hidden_dim * heads)
        
        # Layer 2: Second attention layer
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=2, dropout=dropout)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        
        # Layer 3: Output layer averaging heads
        self.gat3 = GATConv(hidden_dim * 2, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.gat1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.gat2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3
        x = self.gat3(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_attention_weights(self, x: torch.Tensor, edge_index: torch.Tensor):
        """Returns attention coefficients for explainability."""
        self.eval()
        with torch.no_grad():
            _, (edge_index_attn, alpha) = self.gat1(x, edge_index, return_attention_weights=True)
        return edge_index_attn, alpha
