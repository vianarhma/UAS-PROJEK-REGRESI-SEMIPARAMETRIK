"""
Utility functions for the cryptocurrency prediction app
"""

import os
import pickle
import hashlib
from typing import Any, Optional

# Model persistence functions
def save_model(model: Any, model_name: str, model_dir: str = "saved_models", use_timestamp: bool = False) -> str:
    """
    Save a trained model to disk as .pkl file

    Args:
        model: The trained model object
        model_name: Name identifier for the model
        model_dir: Directory to save models
        use_timestamp: If True, adds timestamp to filename. If False, overwrites latest.

    Returns:
        Path to the saved model file
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if use_timestamp:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.pkl"
    else:
        filename = f"{model_name}_latest.pkl"
        
    filepath = os.path.join(model_dir, filename)

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        return filepath
    except Exception as e:
        raise Exception(f"Failed to save model {model_name}: {str(e)}")

def load_latest_model(model_name: str, model_dir: str = "saved_models") -> Optional[Any]:
    """
    Load the latest saved model for a given model name
    Priority: 1. model_name_latest.pkl, 2. most recent timestamped file
    """
    if not os.path.exists(model_dir):
        return None

    # Try fixed latest name first
    latest_fixed = os.path.join(model_dir, f"{model_name}_latest.pkl")
    if os.path.exists(latest_fixed):
        try:
            with open(latest_fixed, 'rb') as f:
                return pickle.load(f)
        except:
            pass # Fallback to search

    # Find all files matching the model name pattern
    model_files = [f for f in os.listdir(model_dir)
                   if f.startswith(model_name) and f.endswith('.pkl')]

    if not model_files:
        return None

    # Sort by timestamp (newest first) and load the latest
    model_files.sort(reverse=True)
    latest_file = os.path.join(model_dir, model_files[0])

    try:
        with open(latest_file, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        print(f"Failed to load model {model_name}: {str(e)}")
        return None

def get_model_hash(model_data: dict) -> str:
    """
    Generate a hash based on model training data to check if retraining is needed

    Args:
        model_data: Dictionary containing training data info

    Returns:
        Hash string representing the model configuration
    """
    # Create a string representation of key parameters
    hash_string = f"{model_data.get('n_samples', '')}_{model_data.get('features', '')}_{model_data.get('target', '')}"
    return hashlib.md5(hash_string.encode()).hexdigest()

# File kosong atau bisa tambahkan:
# from .styling import apply_custom_css, render_header, render_metric_card