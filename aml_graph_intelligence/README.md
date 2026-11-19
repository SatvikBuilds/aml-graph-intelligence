# Dynamic Graph Intelligence for Multi-Hop Anti-Money Laundering Detection

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch%20%26%20PyG-EE4C2C.svg)](https://pytorch.org)
[![NetworkX](https://img.shields.io/badge/Graph%20Engine-NetworkX-green.svg)](https://networkx.org)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An intelligent, enterprise-grade Anti-Money Laundering (AML) surveillance platform that models financial transaction systems as directed graphs and employs **Graph Neural Networks (GNN)** to detect complex, multi-hop financial crime topologies.

---

## 1. Executive Summary & Problem Context

Traditional Anti-Money Laundering (AML) monitoring systems rely heavily on static, rule-based threshold filters (e.g., flagging transactions exceeding $10,000). While effective at catching unsophisticated fraud, these isolated checks are fundamentally incapable of recognizing coordinated, distributed money laundering typologies where illicit proceeds are obscured across multiple intermediate accounts:

1. **Structuring / Smurfing (Fan-In)**: Dividing large illicit sums into dozens of small transactions just below the \$10,000 regulatory reporting threshold, aggregated into a mule account.
2. **Multi-Hop Layering Chains**: Forwarding funds sequentially through 3 to 6 intermediary shell accounts with rapid velocity (15 to 60 minutes) to disrupt the audit trail.
3. **Circular Flow / Round-Tripping**: Channelling capital through circular directed paths ($A \to B \to C \to A$) disguised as legitimate commercial invoices.
4. **Dispersal / Fan-Out**: Rapidly routing aggregated capital into numerous foreign withdrawal nodes.

**Dynamic Graph Intelligence** solves this limitation by constructing a rich directed graph $G = (V, E)$, extracting high-order topological and temporal features, and applying **Graph Convolutional Networks (GCN)** and **Graph Attention Networks (GAT)** to capture relational dependencies across multi-hop payment paths.

---

## 2. Technology Stack

- **Core Language**: Python 3.10+ (tested on Python 3.13)
- **Web Application Framework**: Streamlit
- **Deep Graph Learning**: PyTorch 2.15+ & PyTorch Geometric (PyG) 2.8+
- **GNN Architectures**: Graph Convolutional Network (GCN) & Graph Attention Network (GAT)
- **Graph Modeling & Analytics**: NetworkX 3.5
- **Interactive Visualizations**: PyVis (Physics simulation) & Plotly (2D WebGL)
- **Database Backend**: MongoDB (PyMongo 4.20) for user authentication, investigation cases, and regulatory audit logging
- **Data Engineering**: Pandas, NumPy, Scikit-learn
- **Graph Database Export**: Neo4j Cypher generation script support

---

## 3. Project Architecture

```
aml_graph_intelligence/
│
├── app.py                         # Application entrypoint & architecture hub
├── requirements.txt               # Locked dependencies
├── README.md                      # Comprehensive documentation
│
├── pages/                         # Streamlit multi-page interface
│   ├── 1_Dashboard.py             # Macro KPIs, financial dynamics & alert feed
│   ├── 2_Dataset_Upload.py        # CSV ingestion, schema mapper & quality audit
│   ├── 3_Transaction_Explorer.py  # Filterable ledger & entity forensics
│   ├── 4_Graph_Visualization.py   # Flagship interactive PyVis physics network
│   ├── 5_AI_Detection.py          # GCN/GAT model training, metrics & ROC curves
│   └── 6_Investigation_Center.py  # Forensic dossier & SAR audit export
│
├── preprocessing/
│   ├── data_cleaning.py           # Schema normalization, self-loop pruning, validation
│   └── feature_engineering.py     # Graph centrality, flow ratios & velocity metrics
│
├── graph/
│   ├── graph_builder.py           # NetworkX DiGraph builder & Neo4j Cypher generator
│   └── graph_analysis.py          # K-hop ego traversal, cycles & smurfing detectors
│
├── models/
│   ├── gcn_model.py               # 3-Layer Graph Convolutional Network
│   ├── gat_model.py               # Multi-Head Graph Attention Network
│   ├── train.py                   # Stratified training pipeline & test evaluation
│   └── prediction.py              # Inferences, baseline heuristics & XAI reason codes
│
├── visualization/
│   ├── network_visualization.py   # PyVis interactive HTML & Plotly network
│   └── charts.py                  # Dark-navy Plotly timeline, pie & matrix charts
│
├── utils/
│   ├── helpers.py                 # State management & dark fintech CSS styling
│   └── report_generator.py        # Downloadable CSV & print-ready HTML dossiers
│
└── dataset/
    ├── sample_transactions.csv    # Benchmark multi-hop AML transaction dataset
    └── generate_sample_data.py    # Synthetic AML topology generator script
```

---

## 4. Graph Neural Network Formulations

### Graph Convolutional Network (GCN)
For node representations $H^{(l)}$ at layer $l$, the GCN performs localized spectral graph convolutions:
$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$$
where $\tilde{A} = A + I_N$ is the graph adjacency matrix with added self-loops, $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$ is its diagonal degree matrix, and $W^{(l)}$ is the layer weight matrix.

### Graph Attention Network (GAT)
GAT assigns dynamic attention weights $\alpha_{ij}$ to each directed transfer edge, learning which neighboring accounts carry highest threat contagion:
$$\alpha_{ij} = \frac{\exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \,\|\, \mathbf{W}h_j]\right)\right)}{\sum_{k \in \mathcal{N}_i} \exp\left(\text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}h_i \,\|\, \mathbf{W}h_k]\right)\right)}$$

---

## 5. Installation and Execution

### Step 1: Clone or Navigate to Project Directory
```powershell
cd c:\DGI
```

### Step 2: Install Dependencies
```powershell
pip install -r requirements.txt
```
*(PyTorch and PyTorch Geometric can also be installed using CPU wheels on Windows as documented in requirements).*

### Step 3: Run the Streamlit Application
```powershell
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 6. Page-by-Page Feature Walkthrough

### 1. Executive Dashboard (`1_Dashboard.py`)
- **6 KPI Metric Cards**: Total Transactions, Total Accounts, Suspicious Transactions, High-Risk Accounts, Suspicious Networks Detected, and Cumulative Volume.
- **Dynamic Charts**: Volume over time area chart, legitimate vs suspicious donut chart, risk score distribution histogram, and daily threat trends.
- **Alert Feed**: Real-time ticker of flagged high-risk accounts and transactions.

### 2. Dataset Management (`2_Dataset_Upload.py`)
- **CSV Ingestion**: Upload custom datasets (IBM AML, PaySim, Kaggle, or internal logs).
- **Flexible Schema Mapping**: Interactive dropdowns auto-detect and map varying column headers (`orig`, `from_account`, `trans_date`, etc.).
- **Data Quality Diagnostics**: Row count, column null distributions, self-loop pruning, and sample CSV download.
- **Start Analysis**: One-click pipeline execution that constructs the graph and extracts topological features.

### 3. Transaction Explorer (`3_Transaction_Explorer.py`)
- **High-Performance Filterable Table**: Filter by account ID, risk score range, detection status, date bounds, and transfer amount.
- **Entity Forensic Drawer**: Click to inspect sender/receiver account histories, inflow/outflow volume, and regulatory reason codes.
- **Export**: Instant CSV export of filtered records.

### 4. Graph Network Visualization (`4_Graph_Visualization.py`)
- **Interactive PyVis Physics**: Drag nodes, zoom, pan, hover for transaction tooltips, and follow directed arrows.
- **Color Coding**: Red (High Risk $\ge 0.70$), Yellow (Medium Risk $0.40 - 0.69$), Green (Low Risk $0.05 - 0.39$), Blue (Normal $< 0.05$).
- **Multi-Hop Traversal**: Search any account ID and extract its 1-hop, 2-hop, or 3-hop ego network.
- **Topology Presets**: Highlight detected circular round-trips, structuring smurfs, or multi-hop layering chains with a single click.

### 5. AI Detection Studio (`5_AI_Detection.py`)
- **GNN Model Trainer**: Train GCN or GAT models with selectable epochs, learning rates, and hidden dimensions.
- **Evaluation Metrics**: Test Accuracy, Precision, Recall, F1-Score, ROC-AUC, and annotated Confusion Matrix.
- **Transparent Baseline Mode**: When a deep GNN has not been trained, clearly presents baseline graph heuristics without faking results.

### 6. Investigation Center (`6_Investigation_Center.py`)
- **Compliance Dossier**: Full subject account file with composite threat score, cumulative in/out flow, and velocity metrics.
- **Multi-Hop Fund Tracer**: Find directed flow paths connecting the subject to any counterparty.
- **Workflow Management**: Set case status (`New`, `Under Investigation`, `Resolved`, `Flagged for SAR / STR`) and save investigator notes.
- **Export Reports**: Generate both CSV audits and print-ready HTML investigative dossiers for law enforcement / FIU reporting.

---

## 7. Optional Neo4j Graph Database Integration

The system includes a built-in Neo4j Cypher exporter (`graph/graph_builder.py`). Compliance officers can export graph entities directly into Cypher statements:

```cypher
CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.id IS UNIQUE;
MERGE (s:Account {id: 'ACC_SMURF_01'}) MERGE (r:Account {id: 'ACC_MULE_AGG_01'})
CREATE (s)-[:TRANSFERRED {txn_id: 'TXN_10023', amount: 9850.0, timestamp: '2025-01-18 10:15:00', is_laundering: 1}]->(r);
```

---

## 8. Academic & Forensic Integrity

- **No Fabricated Predictions**: Baseline heuristic scores are explicitly labeled as `Baseline Graph Heuristic`. When GNN models are trained, evaluation is performed strictly on stratified holdout test masks to ensure zero data leakage.
- **Explainability**: Every flagged account includes explainable compliance reason codes identifying the precise topological triggers (velocity, structuring, cycles, network contagion).

---

Developed for College AIML Project & Financial Intelligence Research.
