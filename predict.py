import argparse
import pandas as pd
import json
import os
from src.utils import load_dataset
from src.pseudo_label import generate_pseudo_labels
from src.inference import generate_evidence_dossier

def main():
    parser = argparse.ArgumentParser(description="Support Integrity Auditor Inference")
    parser.add_argument("--input", type=str, required=False, default="data/customer_support_tickets.csv", help="Path to input CSV")
    args = parser.parse_args()
    
    print(f"Running Inference on {args.input}...")
    
    df = load_dataset(args.input)
    
    # In a real scenario, we would run the trained DeBERTa model here.
    # For this skeleton, we use the pseudo-label logic to simulate inference.
    df = generate_pseudo_labels(df)
    
    mismatches = df[df['mismatch_label'] == 1]
    print(f"Detected {len(mismatches)} Priority Mismatches. Generating dossiers...")
    
    dossiers = []
    for _, row in mismatches.iterrows():
        dossier = generate_evidence_dossier(row)
        dossiers.append(dossier)
        
    # Save dossiers
    output_path = "results/dossiers.json"
    os.makedirs("results", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dossiers, f, indent=4)
        
    print(f"Saved {len(dossiers)} dossiers to {output_path}")

if __name__ == "__main__":
    main()
