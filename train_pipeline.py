from src.utils import load_dataset
from src.pseudo_label import generate_pseudo_labels
from src.train import train_model
import os

def main():
    print("Starting Support Integrity Auditor Training Pipeline...")
    # 1. Load data
    data_path = os.path.join("data", "customer_support_tickets.csv")
    df = load_dataset(data_path)
    
    # 2. Generate pseudo labels
    df = generate_pseudo_labels(df)
    
    # 3. Train model
    # We only pass data with valid labels if needed, but our pseudo-labeler labels all.
    trainer, tokenizer = train_model(df)
    
    # 4. Save artifacts explicitly (optional, since trainer saves them)
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
