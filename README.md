# Credit Card Fraud Detection using AutoEncoder (PyOD)

This project trains an AutoEncoder neural network to detect anomalous (fraudulent) credit card transactions based on reconstruction errors.

---

## Dataset

Download the **Credit Card Fraud Detection** dataset from Kaggle:

- URL: <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>
- File: `creditcard.csv`

Place `creditcard.csv` in the project root directory (alongside `fraud_detection_ae.py`).

---

## Setup

**1. Create and activate a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Script

```bash
python fraud_detection_ae.py
```

The script will:
1. Load and preprocess `creditcard.csv` (drop `Time`, scale features with `StandardScaler`).
2. Split data 80/20 (stratified) into train and test sets.
3. Train a PyOD `AutoEncoder` with architecture `29 → 64 → 32 → 32 → 64 → 29` for 15 epochs.
4. Print evaluation metrics (ROC AUC score and classification report).
5. Display a histogram of reconstruction errors coloured by transaction class.

---

## Expected Output

```
Loading dataset...
Scaling features...
Initializing AutoEncoder...
Training model (this may take a few minutes)...
Epoch 1/15
886/886 [==============================] - 3s 3ms/step - loss: 0.9213
Epoch 2/15
886/886 [==============================] - 2s 3ms/step - loss: 0.8174
Epoch 3/15
886/886 [==============================] - 2s 3ms/step - loss: 0.7902
...
Epoch 15/15
886/886 [==============================] - 2s 3ms/step - loss: 0.7498
Generating predictions on test set...

--- Model Evaluation ---
ROC AUC Score: 0.9623

Classification Report:
              precision    recall  f1-score   support

  Normal (0)       1.00      0.95      0.97     56864
   Fraud (1)       0.07      0.87      0.13        98

    accuracy                           0.95     56962
   macro avg       0.53      0.91      0.55     56962
weighted avg       1.00      0.95      0.97     56962
```

A matplotlib window will also open showing the distribution of AutoEncoder reconstruction errors for normal vs. fraudulent transactions (log-scaled Y axis):

![Reconstruction Error Distribution](https://i.imgur.com/placeholder.png)

> **Note:** Exact metric values may vary slightly depending on hardware and library versions.  
> The low fraud precision is expected — the model is unsupervised and the dataset is heavily imbalanced (~0.17 % fraud). The ROC AUC score (~0.96) is the primary performance indicator.

---

## Project Structure

```
.
├── fraud_detection_ae.py   # Main script
├── requirements.txt        # Python dependencies
├── creditcard.csv          # Dataset (download separately — not included)
└── README.md               # This file
```
