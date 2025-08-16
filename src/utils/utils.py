import numpy as np
import joblib
import os
import random

def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across numpy and random.
    """
    np.random.seed(seed)
    random.seed(seed)

def save_object(file_path: str, obj):
    """
    Save a Python object to disk using joblib.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(obj, file_path)

def load_object(file_path: str):
    """
    Load a Python object from disk using joblib.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")
    return joblib.load(file_path)
