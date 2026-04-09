"""
HW2 Part 2: BERT Fine-Tuning & Scaling
========================================
- Model A (Base): bert-base-cased (110M parameters)
- Model B (Large): bert-large-cased (340M parameters)
- Evaluation: ROC-AUC on held-out validation set
- Loss curves & performance comparison
"""

import os
import sys
import pickle
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import Dataset

def main():
    # ================= CONFIGURATION =================
    # Choose model via command-line arg or default to base
    if len(sys.argv) > 1 and sys.argv[1] == "large":
        MODEL_NAME = "bert-large-cased"
        BATCH_SIZE = 16
    else:
        MODEL_NAME = "bert-base-cased"
        BATCH_SIZE = 32

    MAX_LEN = 512
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01

    DATA_SPLIT_PATH = os.path.join("results_baseline", "data_split.pkl")
    OUTPUT_DIR = f"results_{MODEL_NAME}"
    SAVE_DIR = f"saved_model_{MODEL_NAME}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 60)
    print(f"Part 2: BERT Fine-Tuning — {MODEL_NAME}")
    print(f"  Batch Size: {BATCH_SIZE}, Max Length: {MAX_LEN}, Epochs: {EPOCHS}")
    print(f"  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    # =================================================

    # =============================================
    # 1. Load Data Split from Part 1
    # =============================================
    if not os.path.exists(DATA_SPLIT_PATH):
        print(f"ERROR: {DATA_SPLIT_PATH} not found. Run Baseline.py first!")
        sys.exit(1)

    with open(DATA_SPLIT_PATH, 'rb') as f:
        split_data = pickle.load(f)

    X_train = split_data['X_train']
    X_val = split_data['X_val']
    y_train = split_data['y_train']
    y_val = split_data['y_val']

    print(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    # =============================================
    # 2. Build Hugging Face Datasets
    # =============================================
    train_df = pd.DataFrame({'text': X_train, 'label': y_train})
    val_df = pd.DataFrame({'text': X_val, 'label': y_val})

    hf_train = Dataset.from_pandas(train_df.reset_index(drop=True))
    hf_val = Dataset.from_pandas(val_df.reset_index(drop=True))

    # =============================================
    # 3. Tokenization
    # =============================================
    print(f"\nLoading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN)

    print("Tokenizing datasets...")
    tokenized_train = hf_train.map(tokenize_function, batched=True, batch_size=256, desc="Tokenizing train")
    tokenized_val = hf_val.map(tokenize_function, batched=True, batch_size=256, desc="Tokenizing val")

    # Set format for PyTorch
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    # =============================================
    # 4. Model Setup
    # =============================================
    print(f"\nLoading model: {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # =============================================
    # 5. Custom metric computation for Trainer
    # =============================================
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
        preds = np.argmax(logits, axis=-1)
        auc = roc_auc_score(labels, probs)
        acc = accuracy_score(labels, preds)
        return {"roc_auc": auc, "accuracy": acc}

    # =============================================
    # 6. Training Arguments (Optimized for RTX 4090)
    # =============================================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        logging_steps=50,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        save_total_limit=2,
        dataloader_num_workers=0,
        warmup_ratio=0.1,
    )

    # =============================================
    # 7. Train
    # =============================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
    )

    print(f"\nStarting training for {MODEL_NAME}...")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed / 60:.1f} minutes.")

    # =============================================
    # 8. Evaluation
    # =============================================
    print("\nEvaluating on validation set...")
    eval_results = trainer.evaluate()
    print(f"\n{'='*40}")
    print(f"Evaluation Results for {MODEL_NAME}:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"{'='*40}")

    # =============================================
    # 9. Detailed predictions for ROC curve
    # =============================================
    predictions = trainer.predict(tokenized_val)
    logits = predictions.predictions
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    pred_labels = np.argmax(logits, axis=-1)
    val_labels = y_val.values

    auc = roc_auc_score(val_labels, probs)
    acc = accuracy_score(val_labels, pred_labels)

    print(f"\nFinal Validation ROC-AUC: {auc:.4f}")
    print(f"Final Validation Accuracy: {acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(val_labels, pred_labels, target_names=['Human (0)', 'AI (1)']))

    # =============================================
    # 10. Loss Curve Plot
    # =============================================
    log_history = trainer.state.log_history

    train_losses = [(entry['step'], entry['loss']) for entry in log_history if 'loss' in entry]
    eval_losses = [(entry['step'], entry['eval_loss']) for entry in log_history if 'eval_loss' in entry]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    if train_losses:
        steps_tr, losses_tr = zip(*train_losses)
        ax1.plot(steps_tr, losses_tr, label='Train Loss', color='steelblue', alpha=0.7)
    if eval_losses:
        steps_ev, losses_ev = zip(*eval_losses)
        ax1.plot(steps_ev, losses_ev, label='Eval Loss', color='salmon', marker='o')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{MODEL_NAME} — Loss Curve')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ROC Curve
    fpr, tpr, _ = roc_curve(val_labels, probs)
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'{MODEL_NAME} (AUC = {auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'{MODEL_NAME} — ROC Curve')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nCurves saved to {OUTPUT_DIR}/{MODEL_NAME}_curves.png")

    # =============================================
    # 11. Save model, tokenizer, and results
    # =============================================
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)

    results_summary = {
        "model": MODEL_NAME,
        "roc_auc": round(auc, 4),
        "accuracy": round(acc, 4),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "training_time_minutes": round(elapsed / 60, 1),
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LEN,
        "learning_rate": LEARNING_RATE,
    }
    with open(os.path.join(OUTPUT_DIR, "results_summary.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nModel saved to {SAVE_DIR}/")
    print(f"Results saved to {OUTPUT_DIR}/results_summary.json")
    print(f"\n{MODEL_NAME} Part 2 complete!")


if __name__ == '__main__':
    main()