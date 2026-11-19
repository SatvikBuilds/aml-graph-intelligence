"""
Graph Convolutional Network (GCN) Architecture
Uses PyTorch Geometric GCNConv layers for semi-supervised / supervised
node-level money laundering classification.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNClassifier(nn.Module):
    """
    3-Layer Graph Convolutional Network for Account Classification
    """
    def __init__(
        self, 
        in_channels: int, 
        hidden_dim: int = 64, 
        num_classes: int = 2, 
        dropout: float = 0.3
    ):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.conv3 = GCNConv(hidden_dim // 2, num_classes)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Layer 3 (Output logits)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Returns 2nd-layer latent graph representations for visualization."""
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        return x
