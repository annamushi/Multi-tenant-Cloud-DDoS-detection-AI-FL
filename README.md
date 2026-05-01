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

## Key Features

- **Federated Learning Implementation**: Using Flower (flwr) framework for distributed training across multiple tenants
- **Privacy Preservation**: Raw data never leaves the tenant container at any point
- **High Accuracy**: Achieved 99.86% global accuracy using Logistic Regression with FedAvg
- **Non-IID Data Partitioning**: Each tenant holds a unique subset of DDoS attack types
- **Scalability Testing**: Evaluated across 3, 4, and 5 tenant configurations
- **Model Poisoning Robustness**: Tested against label flipping attacks
- **Explainability**: Per-tenant SHAP analysis applied to understand feature importance

## Methodology

### Dataset
The CICDDoS2019 dataset was used, containing labeled network traffic flows across multiple DDoS attack types including DNS, LDAP, MSSQL, NetBIOS, NTP, SNMP, SSDP, UDP, and UDPLag.

### Data Partitioning

- The combined balanced dataset (113,680 samples) was split into three non-IID tenant subsets
- Each tenant was assigned unique attack types to simulate realistic multi-tenant traffic
- Tenants vary in size representing small, medium, and large cloud customers

| Tenant | Attack Types | Total Samples |
|--------|-------------|---------------|
| Tenant 1 (Small) | DNS, LDAP | 12,432 |
| Tenant 2 (Medium) | MSSQL, NetBIOS, NTP | 30,950 |
| Tenant 3 (Large) | UDP, SSDP, SNMP, UDPLag | 62,376 |

### Federated Learning Process

1. **Feature Selection**: Top 15 features selected using Random Forest importance ranking
2. **Local Training**: Each tenant trains a Logistic Regression model on its private data
3. **Model Update**: Only model weights are sent to the Flower FL server
4. **Aggregation**: Server combines weights using FedAvg algorithm
5. **Global Model**: Updated model distributed back to all tenants
6. **Iteration**: Process repeated for 5 rounds until convergence

### Results

| Configuration | Accuracy | Overhead per Round |
|--------------|----------|--------------------|
| 3 Tenants | 99.86% | 0.94 KB |
| 4 Tenants | 99.84% | 1.25 KB |
| 5 Tenants | 99.82% | 1.56 KB |

### Author
Anna Mushi  
MSc Cloud Computing  
Munster Technological University Cork  
