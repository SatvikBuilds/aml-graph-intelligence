"""
Data Cleaning and Validation Module for AML Transaction Data
Supports automated and user-configurable column mapping, timestamp standardization,
and data quality reporting.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any, Optional

STANDARD_COLUMNS = {
    "sender_account": "Sender Account",
    "receiver_account": "Receiver Account",
    "amount": "Transaction Amount",
    "timestamp": "Timestamp",
    "transaction_type": "Transaction Type",
    "is_laundering": "AML / Fraud Label",
    "transaction_id": "Transaction ID"
}

# Alias dictionaries for auto-mapping
COLUMN_ALIASES = {
    "sender_account": [
        "sender", "sender_account", "sender_id", "source", "source_account", 
        "from", "from_account", "from_acc", "orig", "nameorig", "account_from", 
        "src", "originator", "payer"
    ],
    "receiver_account": [
        "receiver", "receiver_account", "receiver_id", "destination", "dest", 
        "target", "to", "to_account", "to_acc", "namedest", "account_to", 
        "dst", "beneficiary", "payee"
    ],
    "amount": [
        "amount", "amt", "transaction_amount", "trans_amount", "value", 
        "tx_amount", "sum", "usd_amount"
    ],
    "timestamp": [
        "timestamp", "date", "trans_date", "time", "datetime", "step", 
        "created_at", "trans_time", "tx_timestamp", "tx_date"
    ],
    "transaction_type": [
        "transaction_type", "type", "txn_type", "trans_type", "category", 
        "payment_type", "action"
    ],
    "is_laundering": [
        "is_laundering", "is_fraud", "isfraud", "is_sar", "label", 
        "fraud", "class", "is_suspicious", "target", "flag"
    ],
    "transaction_id": [
        "transaction_id", "txn_id", "tx_id", "id", "trans_id", "reference"
    ]
}

def detect_column_mapping(df_columns: list) -> Dict[str, Optional[str]]:
    """Automatically maps dataset columns to standard AML schema based on aliases."""
    mapping = {k: None for k in STANDARD_COLUMNS.keys()}
    cleaned_cols = {col: str(col).strip().lower().replace(" ", "_").replace("-", "_") for col in df_columns}
    
    for std_key, aliases in COLUMN_ALIASES.items():
        # First exact clean match
        for orig_col, clean_col in cleaned_cols.items():
            if clean_col in aliases:
                mapping[std_key] = orig_col
                break
        # Substring match if not found
        if mapping[std_key] is None:
            for orig_col, clean_col in cleaned_cols.items():
                for alias in aliases:
                    if alias in clean_col:
                        mapping[std_key] = orig_col
                        break
                if mapping[std_key] is not None:
                    break
                    
    return mapping

def clean_transaction_data(
    df: pd.DataFrame, 
    mapping: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans and standardizes raw transaction dataframe according to schema mapping.
    
    Returns:
        cleaned_df: Standardized dataframe with guaranteed columns
        report: Dictionary with data quality diagnostics
    """
    if df is None or len(df) == 0:
        raise ValueError("Provided dataframe is empty.")

    raw_rows = len(df)
    df_clean = df.copy()

    # Use default detection if mapping not provided
    if not mapping:
        mapping = detect_column_mapping(list(df.columns))

    # Invert mapping for renaming
    rename_dict = {}
    for std_name, orig_col in mapping.items():
        if orig_col and orig_col in df_clean.columns:
            rename_dict[orig_col] = std_name

    df_clean = df_clean.rename(columns=rename_dict)

    # Validate essential minimum columns
    required_cols = ["sender_account", "receiver_account", "amount"]
    missing_required = [col for col in required_cols if col not in df_clean.columns]
    if missing_required:
        raise ValueError(f"Missing mandatory columns: {missing_required}. Please map sender, receiver, and amount.")

    # 1. Clean Account identifiers
    df_clean["sender_account"] = df_clean["sender_account"].astype(str).str.strip()
    df_clean["receiver_account"] = df_clean["receiver_account"].astype(str).str.strip()

    # 2. Clean Amount
    df_clean["amount"] = pd.to_numeric(df_clean["amount"], errors="coerce")
    initial_valid = len(df_clean)
    df_clean = df_clean.dropna(subset=["sender_account", "receiver_account", "amount"])
    df_clean = df_clean[df_clean["amount"] > 0]
    dropped_invalid_amounts = initial_valid - len(df_clean)

    # 3. Handle Transaction ID
    if "transaction_id" not in df_clean.columns or df_clean["transaction_id"].isnull().all():
        df_clean["transaction_id"] = [f"TXN_{10001 + i}" for i in range(len(df_clean))]
    else:
        df_clean["transaction_id"] = df_clean["transaction_id"].astype(str)

    # 4. Standardize Timestamp
    if "timestamp" not in df_clean.columns or df_clean["timestamp"].isnull().all():
        # Synthesize timestamps if completely missing
        base_time = datetime(2025, 1, 1, 9, 0, 0)
        df_clean["timestamp"] = [base_time + timedelta(minutes=15 * i) for i in range(len(df_clean))]
    else:
        # Check if timestamp is numeric step (e.g. PaySim hour step)
        sample_val = df_clean["timestamp"].dropna().iloc[0] if len(df_clean["timestamp"].dropna()) > 0 else 0
        if isinstance(sample_val, (int, float, np.integer, np.floating)) or (isinstance(sample_val, str) and sample_val.isdigit()):
            base_time = datetime(2025, 1, 1, 0, 0, 0)
            df_clean["timestamp"] = pd.to_numeric(df_clean["timestamp"], errors="coerce").fillna(0)
            df_clean["timestamp"] = df_clean["timestamp"].apply(lambda step: base_time + timedelta(hours=float(step)))
        else:
            df_clean["timestamp"] = pd.to_datetime(df_clean["timestamp"], errors="coerce")
            # Fill missing timestamps
            base_time = datetime(2025, 1, 1, 9, 0, 0)
            df_clean["timestamp"] = df_clean["timestamp"].fillna(base_time)

    # Sort by timestamp
    df_clean = df_clean.sort_values(by="timestamp").reset_index(drop=True)

    # 5. Standardize Transaction Type
    if "transaction_type" not in df_clean.columns:
        df_clean["transaction_type"] = "TRANSFER"
    else:
        df_clean["transaction_type"] = df_clean["transaction_type"].fillna("TRANSFER").astype(str).str.upper()

    # 6. Standardize AML / Fraud Label
    has_labels = False
    if "is_laundering" in df_clean.columns:
        df_clean["is_laundering"] = pd.to_numeric(df_clean["is_laundering"], errors="coerce").fillna(0).astype(int)
        # Binarize in case labels are strings or multi-class
        df_clean["is_laundering"] = (df_clean["is_laundering"] > 0).astype(int)
        has_labels = True
    else:
        df_clean["is_laundering"] = 0

    # 7. Remove self-loops where sender == receiver
    self_loops_count = (df_clean["sender_account"] == df_clean["receiver_account"]).sum()
    df_clean = df_clean[df_clean["sender_account"] != df_clean["receiver_account"]].reset_index(drop=True)

    # Data Quality Report
    unique_senders = df_clean["sender_account"].nunique()
    unique_receivers = df_clean["receiver_account"].nunique()
    all_accounts = set(df_clean["sender_account"]).union(set(df_clean["receiver_account"]))
    
    report = {
        "raw_row_count": raw_rows,
        "clean_row_count": len(df_clean),
        "total_accounts": len(all_accounts),
        "unique_senders": unique_senders,
        "unique_receivers": unique_receivers,
        "total_volume": float(df_clean["amount"].sum()),
        "min_amount": float(df_clean["amount"].min()),
        "max_amount": float(df_clean["amount"].max()),
        "mean_amount": float(df_clean["amount"].mean()),
        "date_min": df_clean["timestamp"].min(),
        "date_max": df_clean["timestamp"].max(),
        "dropped_invalid_rows": dropped_invalid_amounts,
        "self_loops_removed": int(self_loops_count),
        "has_labels": has_labels,
        "suspicious_count": int(df_clean["is_laundering"].sum()) if has_labels else 0,
        "suspicious_ratio": float(df_clean["is_laundering"].mean()) if has_labels else 0.0
    }

    return df_clean, report
