"""Implements toy datasets from sklearn."""

import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

def load_diabetes_data():
    # Load diabetes dataset
    X, y = load_diabetes(return_X_y=True)
    # Split dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).T
    X_val = torch.tensor(X_val, dtype=torch.float32).T
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    # Return training and validation datasets
    return X_train, y_train, X_val, y_val