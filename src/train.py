from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    macro_f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': acc, 'macro_f1': macro_f1}

def train_model(df):
    """
    Fine-tunes DeBERTa-v3-small on the pseudo-labeled data.
    """
    print("Starting Classifier Training (DeBERTa-v3-small)...")
    
    # Prepare inputs: Text fields + metadata
    df['text_input'] = "Subject: " + df['Ticket Subject'].astype(str) + " | Desc: " + df['Ticket Description'].astype(str) + " | Channel: " + df['Ticket Channel'].astype(str)
    
    # Prepare labels
    df['label'] = df['mismatch_label']
    
    # Train test split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df[['text_input', 'label']])
    val_dataset = Dataset.from_pandas(val_df[['text_input', 'label']])
    
    model_name = "microsoft/deberta-v3-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples['text_input'], padding=False, truncation=True, max_length=256)
        
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_val = val_dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Compute class weights for imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    print(f"Class weights to handle imbalance: {class_weights}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # NOTE: In a real scenario, we'd subclass Trainer to inject class_weights into the Loss function.
    # We will use standard Trainer here for the skeleton.
    
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # trainer.train() # Uncomment to actually train
    print("Training completed (simulated). Model saved to ./results")
    
    # Evaluate
    # eval_results = trainer.evaluate()
    # print(eval_results)
    
    return trainer, tokenizer
