"""
Credit Card Fraud Detection using PyOD AutoEncoder
This script trains an AutoEncoder neural network to detect anomalous 
(fraudulent) transactions based on reconstruction errors.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from pyod.models.auto_encoder import AutoEncoder

def load_and_preprocess_data(filepath):
    """Loads the dataset and applies standard scaling."""
    print("Loading dataset...")
    df = pd.read_csv(filepath)
    
    # Drop 'Time' as it is a sequential feature not highly relevant for this specific AE architecture
    X = df.drop(columns=['Time', 'Class'])
    y = df['Class']
    
    # Stratified split to maintain the heavily imbalanced ratio of fraud to normal transactions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaling is critical for neural networks and distance-based metrics
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def build_and_train_model(X_train, contamination):
    """Initializes and trains the PyOD AutoEncoder model."""
    print("Initializing AutoEncoder...")
    # Define network architecture: 29 input features -> 64 -> 32 -> 32 -> 64 -> 29 output
    clf = AutoEncoder(
        hidden_neurons=[64, 32, 32, 64],
        epochs=15,          # Number of training iterations
        batch_size=256,     # Number of samples per gradient update
        contamination=contamination, # Expected proportion of outliers
        random_state=42
    )
    
    print("Training model (this may take a few minutes)...")
    clf.fit(X_train)
    return clf

def evaluate_and_visualize(clf, X_test, y_test):
    """Evaluates model performance and generates an output plot for the assignment screenshot."""
    print("Generating predictions on test set...")
    y_test_pred = clf.predict(X_test)  # Binary labels (0: normal, 1: outlier)
    y_test_scores = clf.decision_function(X_test)  # Raw reconstruction errors
    
    # Print metrics
    print("\n--- Model Evaluation ---")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_test_scores):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Normal (0)', 'Fraud (1)']))
    
    # Generate visualization for the assignment screenshot deliverable
    plt.figure(figsize=(10, 6))
    sns.histplot(
        x=y_test_scores, 
        hue=y_test, 
        bins=50, 
        kde=True, 
        palette={0: 'blue', 1: 'red'},
        log_scale=(False, True) # Log scale for Y axis due to class imbalance
    )
    plt.title('Distribution of AutoEncoder Reconstruction Errors')
    plt.xlabel('Reconstruction Error (Anomaly Score)')
    plt.ylabel('Count (Log Scale)')
    plt.legend(title='Transaction Class', labels=['Fraud (1)', 'Normal (0)'])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define dataset path
    DATA_PATH = "creditcard.csv" 
    
    try:
        # 1. Prepare Data
        X_train, X_test, y_train, y_test = load_and_preprocess_data(DATA_PATH)
        
        # Calculate contamination (ratio of fraud in the training set)
        fraud_ratio = y_train.mean()
        
        # 2. Train Model
        model = build_and_train_model(X_train, contamination=fraud_ratio)
        
        # 3. Evaluate & Plot
        evaluate_and_visualize(model, X_test, y_test)
        
    except FileNotFoundError:
        print(f"Error: Could not find '{DATA_PATH}'. Please download the Kaggle dataset and place it in the project root.")
