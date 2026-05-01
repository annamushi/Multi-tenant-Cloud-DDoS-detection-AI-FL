# FL DDoS Detection in Multi-Tenant Cloud Architectures

This project implements a Federated Learning system for real-time DDoS detection in multi-tenant cloud environments using the Flower framework. Each cloud tenant trains a local model on their own private data without sharing it, achieving 99.86% global accuracy while preserving data privacy.

## Project Architecture

```
├── notebooks/
│   └── DDoS_Detection_Analysis.ipynb   # Data processing, feature selection, baseline models, SHAP
├── federated_learning/
│   ├── server.py                        # FL server with FedAvg aggregation
│   ├── client.py                        # FL client with Logistic Regression
│   ├── client_poisoned.py               # Poisoned client for security testing
│   └── utils.py                         # Shared utility functions
├── requirements.txt                     # Required libraries
└── README.md                            # Project documentation
```

## Dataset
This project uses the CICDDoS2019 dataset from the Canadian Institute for Cybersecurity.

Nine attack types were used: DNS, LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, UDP, UDPLag

Dataset link: https://www.unb.ca/cic/datasets/ddos-2019.html

## Tenant Configuration
| Tenant | Attack Types | Total Samples |
|--------|-------------|---------------|
| Tenant 1 (Small) | DNS, LDAP | 12,432 |
| Tenant 2 (Medium) | MSSQL, NetBIOS, NTP | 30,950 |
| Tenant 3 (Large) | UDP, SSDP, SNMP, UDPLag | 62,376 |

## Key Results
| Method | Dataset | Accuracy |
|--------|---------|----------|
| Centralized RF | CICDDoS2019 | 99.97% |
| Centralized XGBoost | CICDDoS2019 | 99.98% |
| FL Logistic Regression | CICDDoS2019 | 99.86% |
| FL XGBoost | CICDDoS2019 | 99.94% |
| FL-DAD (CNN) | CICIDS2017 | 98.70% |
| FedLAD (XGBoost) | CICDDoS2019 | 98.38% |


## How to Run


### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run analysis notebook
Open `notebooks/DDoS_Detection_Analysis.ipynb` in Jupyter Notebook and run all cells.

### 3. Run federated learning
Open four terminals and run:

**Terminal 1 — Start server:**
```bash
python federated_learning/server.py
```

**Terminal 2 — Start Tenant 1:**
```bash
python federated_learning/client.py
```

**Terminal 3 — Start Tenant 2:**
```bash
python federated_learning/client.py
```

**Terminal 4 — Start Tenant 3:**
```bash
python federated_learning/client.py
```

## Author
Anna Mushi  
MSc Cloud Computing  
Munster Technological University Cork  
