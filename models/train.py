"""
GNN Model Training and Evaluation Pipeline
Prepares PyTorch Geometric Data objects, handles class imbalance,
prevents data leakage with stratified splits, and records training history.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, accuracy_score
)
from torch_geometric.data import Data

from models.gcn_model import GCNClassifier
from models.gat_model import GATClassifier

def prepare_pyg_data(
    G, 
    node_df: pd.DataFrame, 
    feature_cols: list,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[Data, StandardScaler, Dict[str, int]]:
    """
    Constructs PyTorch Geometric Data object with node features, edge connectivity,
    and non-leaking stratified train/val/test masks.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Align node ordering
    accounts = list(G.nodes())
    acc_to_idx = {acc: i for i, acc in enumerate(accounts)}
    N = len(accounts)

    # Align features with node order
    node_indexed = node_df.set_index("account_id").reindex(accounts).fillna(0)
    X_raw = node_indexed[feature_cols].values

    # Normalize features
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_raw)
    x = torch.tensor(X_norm, dtype=torch.float32)

    # Extract edge index
    edge_list = list(G.edges())
    if len(edge_list) > 0:
        src_indices = [acc_to_idx[u] for u, v in edge_list]
        dst_indices = [acc_to_idx[v] for u, v in edge_list]
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Labels
    labels = node_indexed["is_suspicious"].values.astype(int)
    y = torch.tensor(labels, dtype=torch.long)

    # Stratified split to prevent data leakage and guarantee fraud nodes in all splits
    fraud_indices = np.where(labels == 1)[0]
    legit_indices = np.where(labels == 0)[0]
    np.random.shuffle(fraud_indices)
    np.random.shuffle(legit_indices)

    def split_indices(indices):
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_part = indices[:n_train]
        val_part = indices[n_train:n_train + n_val]
        test_part = indices[n_train + n_val:]
        return train_part, val_part, test_part

    tr_f, val_f, te_f = split_indices(fraud_indices)
    tr_l, val_l, te_l = split_indices(legit_indices)

    train_idx = np.concatenate([tr_f, tr_l])
    val_idx = np.concatenate([val_f, val_l])
    test_idx = np.concatenate([te_f, te_l])

    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data = Data(
        x=x, 
        edge_index=edge_index, 
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )
    data.accounts = accounts

    return data, scaler, acc_to_idx

def train_gnn_model(
    data: Data, 
    model_type: str = "GCN", 
    epochs: int = 60, 
    lr: float = 0.01, 
    hidden_dim: int = 64,
    callback_fn: Optional[Callable[[int, int, Dict[str, float]], None]] = None
) -> Tuple[nn.Module, Dict[str, list], Dict[str, Any]]:
    """
    Executes training loop for GCN or GAT model.
    
    Returns:
        model: Trained PyTorch GNN model
        history: Epoch-wise loss, accuracy, precision, recall, f1 metrics
        eval_metrics: Final test set performance metrics
    """
    in_channels = data.x.size(1)
    
    # Initialize chosen architecture
    if model_type.upper() == "GAT":
        model = GATClassifier(in_channels=in_channels, hidden_dim=32, heads=4, num_classes=2)
    else:
        model = GCNClassifier(in_channels=in_channels, hidden_dim=hidden_dim, num_classes=2)

    # Class imbalance weighting
    num_legit = int((data.y[data.train_mask] == 0).sum().item())
    num_fraud = int((data.y[data.train_mask] == 1).sum().item())
    ratio = max(1.0, float(num_legit) / max(1.0, float(num_fraud)))
    class_weights = torch.tensor([1.0, ratio], dtype=torch.float32)
    loss_fn = nn.NLLLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_precision": [],
        "val_recall": [],
        "val_acc": []
    }

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        # --- Training step ---
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # --- Validation step ---
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            v_loss = loss_fn(val_out[data.val_mask], data.y[data.val_mask]).item()
            
            val_preds = val_out[data.val_mask].argmax(dim=1).cpu().numpy()
            val_targets = data.y[data.val_mask].cpu().numpy()

            v_acc = accuracy_score(val_targets, val_preds)
            v_prec = precision_score(val_targets, val_preds, zero_division=0)
            v_rec = recall_score(val_targets, val_preds, zero_division=0)
            v_f1 = f1_score(val_targets, val_preds, zero_division=0)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(loss.item(), 4))
        history["val_loss"].append(round(v_loss, 4))
        history["val_f1"].append(round(v_f1, 4))
        history["val_precision"].append(round(v_prec, 4))
        history["val_recall"].append(round(v_rec, 4))
        history["val_acc"].append(round(v_acc, 4))

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}

        if callback_fn:
            callback_fn(epoch, epochs, {
                "train_loss": loss.item(),
                "val_loss": v_loss,
                "val_f1": v_f1
            })

    # Restore best weights
    if best_weights:
        model.load_state_dict(best_weights)

    # --- Final Test Evaluation ---
    model.eval()
    with torch.no_grad():
        test_out = model(data.x, data.edge_index)
        probs = torch.exp(test_out)[:, 1].cpu().numpy()
        test_preds = test_out[data.test_mask].argmax(dim=1).cpu().numpy()
        test_targets = data.y[data.test_mask].cpu().numpy()
        test_probs = probs[data.test_mask.cpu().numpy()]

        test_acc = float(accuracy_score(test_targets, test_preds))
        test_prec = float(precision_score(test_targets, test_preds, zero_division=0))
        test_rec = float(recall_score(test_targets, test_preds, zero_division=0))
        test_f1 = float(f1_score(test_targets, test_preds, zero_division=0))
        
        try:
            test_auc = float(roc_auc_score(test_targets, test_probs))
        except Exception:
            test_auc = 0.5

        cm = confusion_matrix(test_targets, test_preds).tolist()

    eval_metrics = {
        "model_type": model_type.upper(),
        "accuracy": round(test_acc, 4),
        "precision": round(test_prec, 4),
        "recall": round(test_rec, 4),
        "f1_score": round(test_f1, 4),
        "roc_auc": round(test_auc, 4),
        "confusion_matrix": cm,
        "epochs_trained": epochs,
        "test_samples": len(test_targets),
        "fraud_in_test": int(sum(test_targets))
    }

    return model, history, eval_metrics
