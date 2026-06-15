# Support Integrity Auditor (SIA)

A semantics-driven, evidence-grounded automated auditor that detects Priority Mismatch in support tickets.

## Architecture & Methodology
The pipeline consists of three main stages:
1. **Pseudo-label Generation (Self-Supervised)**: Fuses rule-based keyword density scoring and resolution-time regression to generate a reliable mismatch label.
2. **Classifier Training (Fine-Tuned Model)**: A `DeBERTa-v3-small` model is fine-tuned on the pseudo-labels. Class weights are applied to handle label imbalance.
3. **Evidence Dossier Generation**: For any ticket classified as a mismatch, a JSON dossier is generated linking the model's decision back to specific words or metadata constraints, ensuring zero hallucinations.

## Setup Instructions
1. Install dependencies: `pip install -r requirements.txt`
2. Place the Kaggle CRM dataset inside the `data/` directory.

## Running the Project
* **Training Pipeline**: `python train_pipeline.py`
* **Inference**: `python predict.py --input data/customer_support_tickets.csv`
* **Web App**: `streamlit run app.py`

## Metrics
The final fine-tuned model targets the following evaluation thresholds:
- Binary Classification Accuracy ≥ 83%
- Macro F1 Score ≥ 0.82
- Per-Class Recall ≥ 0.78

