"""
Synthetic AML Transaction Dataset Generator
Creates realistic financial networks with legitimate transactions and distinct
multi-hop money laundering typologies:
  1. Smurfing / Structuring (Fan-In)
  2. Multi-Hop Layering Chains (Pass-through hops)
  3. Circular Flow / Round-Tripping (Directed cycles)
  4. Dispersal / Layering (Fan-Out)
  5. Scatter-Gather / Bipartite Layering Networks
  6. High-velocity Rapid Transfers
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_aml_dataset(
    num_legit_accounts: int = 180,
    num_legit_txns: int = 1400,
    seed: int = 42,
    output_path: str = None
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = datetime(2025, 1, 15, 8, 0, 0)
    txns = []
    txn_counter = 10001

    # ==========================================
    # 1. LEGITIMATE TRANSACTIONS
    # ==========================================
    legit_accounts = [f"ACC_LEGIT_{i:04d}" for i in range(1, num_legit_accounts + 1)]
    merchants = [f"MERCHANT_{i:02d}" for i in range(1, 15)]
    employers = [f"CORP_PAYROLL_{i:02d}" for i in range(1, 6)]
    
    all_normal_nodes = legit_accounts + merchants + employers
    
    # Normal P2P, payroll, and merchant payments
    for _ in range(num_legit_txns):
        coin = random.random()
        if coin < 0.25:
            # Payroll
            sender = random.choice(employers)
            receiver = random.choice(legit_accounts)
            amount = round(random.uniform(2500, 8500), 2)
            txn_type = "SALARY"
        elif coin < 0.65:
            # Merchant payment
            sender = random.choice(legit_accounts)
            receiver = random.choice(merchants)
            amount = round(random.uniform(15, 450), 2)
            txn_type = "PAYMENT"
        else:
            # Regular P2P transfer
            sender = random.choice(legit_accounts)
            receiver = random.choice(legit_accounts)
            if sender == receiver:
                receiver = random.choice(merchants)
            amount = round(random.uniform(50, 2200), 2)
            txn_type = random.choice(["TRANSFER", "WIRE", "CHEQUE"])
            
        time_offset_seconds = random.randint(0, 30 * 24 * 3600)  # over 30 days
        timestamp = start_time + timedelta(seconds=time_offset_seconds)
        
        txns.append({
            "transaction_id": f"TXN_{txn_counter}",
            "sender_account": sender,
            "receiver_account": receiver,
            "amount": amount,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_type": txn_type,
            "is_laundering": 0,
            "laundering_type": "Legitimate",
            "risk_score": round(random.uniform(0.01, 0.28), 3)
        })
        txn_counter += 1

    # ==========================================
    # 2. SUSPICIOUS PATTERN: SMURFING (FAN-IN)
    # Multiple small deposits just below threshold into aggregator mule
    # ==========================================
    num_smurf_clusters = 4
    for sc in range(1, num_smurf_clusters + 1):
        aggregator = f"ACC_MULE_AGG_{sc:02d}"
        smurf_senders = [f"ACC_SMURF_{sc:02d}_{i:02d}" for i in range(1, 9)]
        cluster_base_time = start_time + timedelta(days=random.randint(2, 25), hours=random.randint(9, 18))
        
        for idx, smurf in enumerate(smurf_senders):
            # Structuring: amounts just below $10,000 or rounded amounts
            amount = round(random.uniform(8500, 9950), 2)
            # Rapid sequential arrival within minutes/hours
            tx_time = cluster_base_time + timedelta(minutes=random.randint(5, 45) * idx)
            txns.append({
                "transaction_id": f"TXN_{txn_counter}",
                "sender_account": smurf,
                "receiver_account": aggregator,
                "amount": amount,
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_type": "CASH_DEPOSIT",
                "is_laundering": 1,
                "laundering_type": "Fan-In (Smurfing)",
                "risk_score": round(random.uniform(0.85, 0.98), 3)
            })
            txn_counter += 1

    # ==========================================
    # 3. SUSPICIOUS PATTERN: MULTI-HOP LAYERING CHAINS
    # A -> B -> C -> D -> E with rapid pass-through and slight decay (cut)
    # ==========================================
    num_chains = 5
    for c in range(1, num_chains + 1):
        chain_len = random.randint(4, 6)
        chain_nodes = [f"ACC_CHAIN_{c:02d}_HOP_{h}" for h in range(chain_len)]
        chain_start_time = start_time + timedelta(days=random.randint(3, 22), hours=random.randint(8, 14))
        
        curr_amount = round(random.uniform(75000, 250000), 2)
        curr_time = chain_start_time
        
        for h in range(chain_len - 1):
            src = chain_nodes[h]
            dst = chain_nodes[h + 1]
            # Rapid pass-through: 15 to 90 minutes
            curr_time = curr_time + timedelta(minutes=random.randint(15, 90))
            # 1.5% to 3% fee cut per hop
            fee = curr_amount * random.uniform(0.015, 0.03)
            hop_amount = round(curr_amount - fee, 2)
            
            txns.append({
                "transaction_id": f"TXN_{txn_counter}",
                "sender_account": src,
                "receiver_account": dst,
                "amount": hop_amount,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_type": "WIRE",
                "is_laundering": 1,
                "laundering_type": "Layering Chain",
                "risk_score": round(random.uniform(0.88, 0.99), 3)
            })
            txn_counter += 1
            curr_amount = hop_amount

    # ==========================================
    # 4. SUSPICIOUS PATTERN: CIRCULAR FLOW / ROUND-TRIPPING
    # A -> B -> C -> D -> A (Directed Cycle)
    # ==========================================
    num_cycles = 4
    for cy in range(1, num_cycles + 1):
        cycle_len = random.randint(3, 4)
        cycle_nodes = [f"ACC_CYCLE_{cy:02d}_NODE_{n}" for n in range(cycle_len)]
        cycle_start_time = start_time + timedelta(days=random.randint(5, 20), hours=random.randint(10, 16))
        
        base_amount = round(random.uniform(120000, 350000), 2)
        curr_time = cycle_start_time
        
        for i in range(cycle_len):
            src = cycle_nodes[i]
            dst = cycle_nodes[(i + 1) % cycle_len]
            curr_time = curr_time + timedelta(hours=random.randint(2, 8))
            # subtle variance to look like business trade invoices
            amt = round(base_amount * random.uniform(0.97, 1.02), 2)
            
            txns.append({
                "transaction_id": f"TXN_{txn_counter}",
                "sender_account": src,
                "receiver_account": dst,
                "amount": amt,
                "timestamp": curr_time.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_type": "WIRE",
                "is_laundering": 1,
                "laundering_type": "Circular Flow",
                "risk_score": round(random.uniform(0.90, 0.99), 3)
            })
            txn_counter += 1

    # ==========================================
    # 5. SUSPICIOUS PATTERN: FAN-OUT (DISPERSAL)
    # Aggregator receives funds and rapidly disperses to multiple mules
    # ==========================================
    num_fanout = 3
    for fo in range(1, num_fanout + 1):
        source_mule = f"ACC_MULE_AGG_{fo:02d}"  # Connects back to smurfing aggregator!
        dest_mules = [f"ACC_OUT_MULE_{fo:02d}_{m:02d}" for m in range(1, 6)]
        fo_time = start_time + timedelta(days=random.randint(10, 24), hours=random.randint(14, 20))
        
        for idx, dm in enumerate(dest_mules):
            amt = round(random.uniform(12000, 19500), 2)
            tx_time = fo_time + timedelta(minutes=random.randint(10, 50) * idx)
            txns.append({
                "transaction_id": f"TXN_{txn_counter}",
                "sender_account": source_mule,
                "receiver_account": dm,
                "amount": amt,
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_type": "TRANSFER",
                "is_laundering": 1,
                "laundering_type": "Fan-Out (Dispersal)",
                "risk_score": round(random.uniform(0.82, 0.96), 3)
            })
            txn_counter += 1

    # ==========================================
    # 6. SUSPICIOUS PATTERN: SCATTER-GATHER (BIPARTITE)
    # Multiple senders -> Intermediary layer -> Multiple exit accounts
    # ==========================================
    inter_layer = ["ACC_SCATTER_MID_1", "ACC_SCATTER_MID_2", "ACC_SCATTER_MID_3"]
    gather_exits = ["ACC_EXIT_OFFSHORE_1", "ACC_EXIT_OFFSHORE_2"]
    scatter_sources = ["ACC_COLLECT_1", "ACC_COLLECT_2", "ACC_COLLECT_3", "ACC_COLLECT_4"]
    
    sg_time = start_time + timedelta(days=18, hours=11)
    for src in scatter_sources:
        for mid in inter_layer:
            amt = round(random.uniform(14000, 28000), 2)
            tx_time = sg_time + timedelta(minutes=random.randint(5, 120))
            txns.append({
                "transaction_id": f"TXN_{txn_counter}",
                "sender_account": src,
                "receiver_account": mid,
                "amount": amt,
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_type": "WIRE",
                "is_laundering": 1,
                "laundering_type": "Scatter-Gather Network",
                "risk_score": round(random.uniform(0.86, 0.97), 3)
            })
            txn_counter += 1

    for mid in inter_layer:
        for exit_acc in gather_exits:
            amt = round(random.uniform(35000, 65000), 2)
            tx_time = sg_time + timedelta(hours=random.randint(2, 6))
            txns.append({
                "transaction_id": f"TXN_{txn_counter}",
                "sender_account": mid,
                "receiver_account": exit_acc,
                "amount": amt,
                "timestamp": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
                "transaction_type": "WIRE",
                "is_laundering": 1,
                "laundering_type": "Scatter-Gather Network",
                "risk_score": round(random.uniform(0.89, 0.98), 3)
            })
            txn_counter += 1

    df = pd.DataFrame(txns)
    # Sort chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Sample dataset generated at: {output_path} ({len(df)} rows)")

    return df

if __name__ == "__main__":
    out_file = os.path.join(os.path.dirname(__file__), "sample_transactions.csv")
    generate_aml_dataset(output_path=out_file)
