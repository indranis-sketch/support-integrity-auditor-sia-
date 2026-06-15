import pandas as pd
import os

def load_dataset(filepath):
    """
    Loads and preprocesses the support ticket dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.read_csv(filepath)
    # Basic preprocessing
    df.fillna('', inplace=True)
    return df
