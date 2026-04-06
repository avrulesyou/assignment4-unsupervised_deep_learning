# Credit Card Fraud Detection — Unsupervised Deep Learning (Assignment 4)

An unsupervised anomaly-detection pipeline that uses a **PyOD AutoEncoder** neural network to identify fraudulent credit card transactions purely from reconstruction error — no labelled training data required.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [How It Works](#how-it-works)
3. [Repository Structure](#repository-structure)
4. [Requirements](#requirements)
5. [Setup](#setup)
6. [Dataset](#dataset)
7. [Running the Script](#running-the-script)
8. [Expected Output](#expected-output)
9. [Model Architecture](#model-architecture)

---

## Project Overview

Credit card fraud is rare (< 0.2 % of transactions) but highly costly. Because labelled fraud examples are scarce and constantly evolving, **unsupervised** methods are well suited to the problem.

This project trains an **AutoEncoder** — a neural network that learns to compress and reconstruct *normal* transactions. Fraudulent transactions, being structurally different, produce a higher **reconstruction error** and are flagged as anomalies.

---

## How It Works

```
Raw Transactions (CSV)
        │
        ▼
  Feature Scaling (StandardScaler)
        │
        ▼
  AutoEncoder Training (on all data, contamination-aware)
        │   Input (29) → [64] → [32] → [32] → [64] → Output (29)
        ▼
  Reconstruction Error per transaction
        │
        ▼
  Threshold (contamination %)  ──►  Normal / Fraud label
        │
        ▼
  Evaluation: ROC AUC + Classification Report + Plot
```

1. **Load & preprocess** – drop the `Time` column, apply `StandardScaler`.
2. **Stratified split** – 80 % train / 20 % test, preserving the class ratio.
3. **Train AutoEncoder** – 15 epochs, batch size 256, hidden layers `[64, 32, 32, 64]`.
4. **Predict** – transactions whose reconstruction error exceeds the contamination threshold are labelled fraud.
5. **Evaluate** – ROC AUC, precision/recall/F1, and a histogram of anomaly scores.

---

## Repository Structure

```
assignment4-unsupervised_deep_learning/
├── fraud_detection_ae.py      # Main script
├── creditcard_sample.csv      # Sample dataset (500 rows) for quick testing
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Requirements

| Package | Version |
|---|---|
| Python | ≥ 3.9 |
| pandas | 2.2.0 |
| numpy | 1.26.4 |
| scikit-learn | 1.4.0 |
| pyod | 1.1.3 |
| tensorflow | 2.15.0 |
| matplotlib | 3.8.2 |
| seaborn | 0.13.2 |

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/avrulesyou/assignment4-unsupervised_deep_learning.git
cd assignment4-unsupervised_deep_learning

# 2. (Recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset

### Option A — Sample dataset (quick start, included in repo)

A 500-row sample CSV (`creditcard_sample.csv`) is included so you can run the script immediately without any downloads. It contains 490 normal and 10 fraudulent transactions with the same column schema as the full dataset.

To use it, update the `DATA_PATH` constant in `fraud_detection_ae.py`:

```python
DATA_PATH = "creditcard_sample.csv"
```

### Option B — Full Kaggle dataset (recommended for real evaluation)

The full **Credit Card Fraud Detection** dataset (~144 MB, 284 807 transactions) is available on Kaggle:

1. Go to <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
2. Download `creditcard.csv` and place it in the project root.
3. Ensure `DATA_PATH = "creditcard.csv"` in `fraud_detection_ae.py` (this is the default).

#### Dataset schema

| Column | Description |
|---|---|
| `Time` | Seconds elapsed since first transaction (dropped during preprocessing) |
| `V1` – `V28` | PCA-transformed anonymised features |
| `Amount` | Transaction amount in EUR |
| `Class` | `0` = legitimate, `1` = fraudulent |

---

## Running the Script

```bash
python fraud_detection_ae.py
```

Console output will look like:

```
Loading dataset...
Scaling features...
Initializing AutoEncoder...
Training model (this may take a few minutes)...
Generating predictions on test set...

--- Model Evaluation ---
ROC AUC Score: 0.9XXX

Classification Report:
              precision    recall  f1-score   support

  Normal (0)       1.00      0.97      0.98     56863
   Fraud (1)       0.XX      0.XX      0.XX        99
```

A histogram showing the distribution of reconstruction errors for normal vs. fraudulent transactions will also be displayed.

---

## Expected Output

| Metric | Typical value (full dataset) |
|---|---|
| ROC AUC | ~0.94 – 0.97 |
| Fraud recall | ~0.60 – 0.75 |
| Fraud precision | ~0.10 – 0.30 |

> **Note:** Precision is low because anomaly detection uses a global reconstruction-error threshold — some normal transactions also score high. The ROC AUC is the more meaningful metric for this task.

---

## Model Architecture

```
Input layer  :  29 neurons  (V1–V28 + Amount)
Hidden layer :  64 neurons  (ReLU)
Bottleneck   :  32 neurons  (ReLU)
Hidden layer :  32 neurons  (ReLU)
Hidden layer :  64 neurons  (ReLU)
Output layer :  29 neurons  (Sigmoid)

Loss: Mean Squared Error
Optimizer: Adam
```

The bottleneck forces the network to learn a compressed representation of *normal* transaction patterns. Anomalous inputs cannot be reconstructed accurately, producing a high MSE that serves as the anomaly score.
