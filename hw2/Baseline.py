"""
HW2 Part 1: Data Analysis & Baseline
=====================================
- Exploratory Data Analysis (EDA): word count distribution, vocabulary richness
- Classic Baseline: TF-IDF + Logistic Regression
- Report: ROC-AUC score as benchmark
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve
import pickle
import re

# ================= CONFIGURATION =================
DATA_PATH = os.path.join("DAIGT V2 Train Dataset", "train_v2_drcat_02.csv")
OUTPUT_DIR = "results_baseline"
RANDOM_STATE = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)
# =================================================

# =============================================
# 1. Load Data
# =============================================
print("=" * 60)
print("Part 1: Data Analysis & Baseline")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
df = df[['text', 'label']].dropna(subset=['text', 'label'])
df['label'] = df['label'].astype(int)

print(f"\nDataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"  Label 0 (Human): {(df['label'] == 0).sum()}")
print(f"  Label 1 (AI):    {(df['label'] == 1).sum()}")

# =============================================
# 2. EDA – Exploratory Data Analysis
# =============================================
print("\n" + "=" * 60)
print("EDA: Exploratory Data Analysis")
print("=" * 60)

# --- Word count ---
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

human_wc = df[df['label'] == 0]['word_count']
ai_wc = df[df['label'] == 1]['word_count']

print(f"\nWord Count Statistics:")
print(f"  Human – mean: {human_wc.mean():.1f}, median: {human_wc.median():.1f}, std: {human_wc.std():.1f}")
print(f"  AI    – mean: {ai_wc.mean():.1f}, median: {ai_wc.median():.1f}, std: {ai_wc.std():.1f}")

# --- Vocabulary richness (Type-Token Ratio) ---
def type_token_ratio(text):
    words = str(text).lower().split()
    if len(words) == 0:
        return 0.0
    return len(set(words)) / len(words)

df['ttr'] = df['text'].apply(type_token_ratio)

human_ttr = df[df['label'] == 0]['ttr']
ai_ttr = df[df['label'] == 1]['ttr']

print(f"\nVocabulary Richness (Type-Token Ratio):")
print(f"  Human – mean: {human_ttr.mean():.4f}, median: {human_ttr.median():.4f}")
print(f"  AI    – mean: {ai_ttr.mean():.4f}, median: {ai_ttr.median():.4f}")

# --- Sentence count ---
def count_sentences(text):
    return len(re.split(r'[.!?]+', str(text)))

df['sentence_count'] = df['text'].apply(count_sentences)

human_sc = df[df['label'] == 0]['sentence_count']
ai_sc = df[df['label'] == 1]['sentence_count']

print(f"\nSentence Count Statistics:")
print(f"  Human – mean: {human_sc.mean():.1f}, median: {human_sc.median():.1f}")
print(f"  AI    – mean: {ai_sc.mean():.1f}, median: {ai_sc.median():.1f}")

# --- Average word length ---
def avg_word_length(text):
    words = str(text).split()
    if len(words) == 0:
        return 0.0
    return np.mean([len(w) for w in words])

df['avg_word_len'] = df['text'].apply(avg_word_length)

human_awl = df[df['label'] == 0]['avg_word_len']
ai_awl = df[df['label'] == 1]['avg_word_len']

print(f"\nAverage Word Length:")
print(f"  Human – mean: {human_awl.mean():.2f}")
print(f"  AI    – mean: {ai_awl.mean():.2f}")

# =============================================
# 3. EDA Plots
# =============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EDA: Human vs AI Text Comparison", fontsize=16, fontweight='bold')

# (a) Word Count Distribution
axes[0, 0].hist(human_wc, bins=50, alpha=0.6, label='Human (0)', color='steelblue', density=True)
axes[0, 0].hist(ai_wc, bins=50, alpha=0.6, label='AI (1)', color='salmon', density=True)
axes[0, 0].set_title('Word Count Distribution')
axes[0, 0].set_xlabel('Word Count')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()

# (b) Type-Token Ratio Distribution
axes[0, 1].hist(human_ttr, bins=50, alpha=0.6, label='Human (0)', color='steelblue', density=True)
axes[0, 1].hist(ai_ttr, bins=50, alpha=0.6, label='AI (1)', color='salmon', density=True)
axes[0, 1].set_title('Vocabulary Richness (Type-Token Ratio)')
axes[0, 1].set_xlabel('TTR')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# (c) Sentence Count Distribution
axes[1, 0].hist(human_sc, bins=50, alpha=0.6, label='Human (0)', color='steelblue', density=True)
axes[1, 0].hist(ai_sc, bins=50, alpha=0.6, label='AI (1)', color='salmon', density=True)
axes[1, 0].set_title('Sentence Count Distribution')
axes[1, 0].set_xlabel('Sentence Count')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()

# (d) Average Word Length Distribution
axes[1, 1].hist(human_awl, bins=50, alpha=0.6, label='Human (0)', color='steelblue', density=True)
axes[1, 1].hist(ai_awl, bins=50, alpha=0.6, label='AI (1)', color='salmon', density=True)
axes[1, 1].set_title('Average Word Length Distribution')
axes[1, 1].set_xlabel('Avg Word Length')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "eda_distributions.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"\nEDA plots saved to {OUTPUT_DIR}/eda_distributions.png")

# =============================================
# 4. Train/Validation Split
# =============================================
print("\n" + "=" * 60)
print("TF-IDF + Logistic Regression Baseline")
print("=" * 60)

X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=RANDOM_STATE, stratify=df['label']
)

print(f"Train set: {len(X_train)} samples")
print(f"Val set:   {len(X_val)} samples")

# =============================================
# 5. TF-IDF + Logistic Regression
# =============================================
print("\nTraining TF-IDF + Logistic Regression...")
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)

clf = LogisticRegression(solver='liblinear', max_iter=1000, C=1.0)
clf.fit(X_train_tfidf, y_train)

# Predictions
preds_proba = clf.predict_proba(X_val_tfidf)[:, 1]
preds_label = clf.predict(X_val_tfidf)

# =============================================
# 6. Evaluation
# =============================================
auc = roc_auc_score(y_val, preds_proba)
acc = accuracy_score(y_val, preds_label)

print(f"\n{'='*40}")
print(f"Baseline TF-IDF + LR Results:")
print(f"  ROC-AUC:  {auc:.4f}")
print(f"  Accuracy: {acc:.4f}")
print(f"{'='*40}")
print(f"\nClassification Report:")
print(classification_report(y_val, preds_label, target_names=['Human (0)', 'AI (1)']))

# =============================================
# 7. ROC Curve Plot
# =============================================
fpr, tpr, _ = roc_curve(y_val, preds_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'TF-IDF + LR (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Baseline ROC Curve: TF-IDF + Logistic Regression', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "baseline_roc_curve.png"), dpi=150, bbox_inches='tight')
plt.close()
print(f"ROC curve saved to {OUTPUT_DIR}/baseline_roc_curve.png")

# =============================================
# 8. Save split and model for BERT scripts
# =============================================
split_data = {
    'X_train': X_train.reset_index(drop=True),
    'X_val': X_val.reset_index(drop=True),
    'y_train': y_train.reset_index(drop=True),
    'y_val': y_val.reset_index(drop=True),
}
with open(os.path.join(OUTPUT_DIR, "data_split.pkl"), 'wb') as f:
    pickle.dump(split_data, f)

print(f"Data split saved to {OUTPUT_DIR}/data_split.pkl")
print(f"\nBaseline ROC-AUC = {auc:.4f} — this is the benchmark for BERT models.")
print("Part 1 complete!")