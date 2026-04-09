"""
HW2 Part 3: Adversarial Attack with Local LLM
===============================================
- Deploy local LLM via Ollama (deepseek-r1:8b)
- Rewrite human essays to fool the BERT detector
- Test defenses: feed attack samples into the best BERT classifier
"""

import os
import sys
import json
import pickle
import time
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import ollama

# ================= CONFIGURATION =================
# Use the best BERT model from Part 2
# Check which model directories exist
if os.path.exists("saved_model_bert-large-cased"):
    DETECTOR_PATH = "saved_model_bert-large-cased"
elif os.path.exists("saved_model_bert-base-cased"):
    DETECTOR_PATH = "saved_model_bert-base-cased"
else:
    print("ERROR: No saved BERT model found. Run BERT.py first!")
    sys.exit(1)

# Ollama model (already pulled)
GEN_MODEL_NAME = "deepseek-r1:8b-llama-distill-fp16"

DATA_SPLIT_PATH = os.path.join("results_baseline", "data_split.pkl")
OUTPUT_DIR = "results_adversarial"
NUM_ATTACK_SAMPLES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
# =================================================

print("=" * 60)
print("Part 3: Adversarial Attack with Local LLM")
print(f"  Detector: {DETECTOR_PATH}")
print(f"  Generator: {GEN_MODEL_NAME} (Ollama)")
print("=" * 60)

# =============================================
# 1. Load validation data
# =============================================
if not os.path.exists(DATA_SPLIT_PATH):
    print(f"ERROR: {DATA_SPLIT_PATH} not found. Run Baseline.py first!")
    sys.exit(1)

with open(DATA_SPLIT_PATH, 'rb') as f:
    split_data = pickle.load(f)

X_val = split_data['X_val']
y_val = split_data['y_val']

# Select human-written essays only (label=0) for rewriting
human_indices = y_val[y_val == 0].index.tolist()
np.random.seed(42)
selected_indices = np.random.choice(human_indices, size=min(NUM_ATTACK_SAMPLES, len(human_indices)), replace=False)
selected_essays = X_val.iloc[selected_indices].tolist()

print(f"\nSelected {len(selected_essays)} human-written essays for adversarial rewriting.")

# =============================================
# 2. Load BERT Detector
# =============================================
print(f"\nLoading BERT detector from {DETECTOR_PATH}...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = AutoModelForSequenceClassification.from_pretrained(DETECTOR_PATH).to(device)
detector.eval()
detector_tokenizer = AutoTokenizer.from_pretrained(DETECTOR_PATH)
print(f"Detector loaded on {device}")


def predict_text(text, max_length=512):
    """Predict whether text is human (0) or AI (1), return probabilities."""
    inputs = detector_tokenizer(
        text, return_tensors="pt", truncation=True,
        max_length=max_length, padding=True
    ).to(device)
    with torch.no_grad():
        logits = detector(**inputs).logits
        probs = torch.softmax(logits, dim=1)
    return probs[0].cpu().numpy()  # [p_human, p_ai]


# =============================================
# 3. Test original human essays against detector
# =============================================
print("\n--- Testing ORIGINAL human essays against detector ---")
original_results = []
for i, essay in enumerate(selected_essays):
    probs = predict_text(essay)
    pred_label = "AI" if probs[1] > 0.5 else "Human"
    original_results.append({
        "index": int(selected_indices[i]),
        "ai_prob": float(probs[1]),
        "pred_label": pred_label,
    })
    print(f"  Essay {i+1}: Pred={pred_label}, AI_prob={probs[1]:.4f}")

# =============================================
# 4. Generate adversarial rewrites using local LLM
# =============================================
print(f"\n--- Generating adversarial rewrites with {GEN_MODEL_NAME} ---")

rewrite_prompts = [
    "Rewrite the following essay to make it sound like it was written by a high school student. Keep the main ideas but change the wording and style:\n\n",
    "Improve the writing style of the following essay while keeping it natural and conversational:\n\n",
    "Rewrite this essay with better vocabulary and sentence structure, but keep it sounding authentic and personal:\n\n",
]

attack_results = []

for i, essay in enumerate(selected_essays):
    prompt_idx = i % len(rewrite_prompts)
    prompt_text = rewrite_prompts[prompt_idx] + essay

    print(f"\n  Rewriting essay {i+1}/{len(selected_essays)} (prompt style {prompt_idx+1})...")
    start_time = time.time()

    try:
        response = ollama.chat(
            model=GEN_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful writing assistant. Rewrite the essay as instructed. Output ONLY the rewritten essay, no explanations or preamble."},
                {"role": "user", "content": prompt_text},
            ],
            options={"temperature": 0.7, "num_predict": 1024},
        )
        generated_text = response.message.content.strip()
        elapsed = time.time() - start_time
        print(f"    Generated {len(generated_text.split())} words in {elapsed:.1f}s")
    except Exception as e:
        print(f"    ERROR generating: {e}")
        generated_text = ""
        elapsed = 0

    if not generated_text:
        print(f"    Skipping empty generation.")
        continue

    # Test against detector
    probs_orig = predict_text(essay)
    probs_attack = predict_text(generated_text)

    pred_orig = "AI" if probs_orig[1] > 0.5 else "Human"
    pred_attack = "AI" if probs_attack[1] > 0.5 else "Human"
    fooled = probs_attack[1] < 0.5  # Detector thinks it's human

    result = {
        "essay_idx": int(selected_indices[i]),
        "prompt_style": prompt_idx + 1,
        "original_snippet": essay[:200],
        "generated_snippet": generated_text[:200],
        "generated_full": generated_text,
        "original_ai_prob": float(probs_orig[1]),
        "original_pred": pred_orig,
        "attack_ai_prob": float(probs_attack[1]),
        "attack_pred": pred_attack,
        "fooled_detector": fooled,
        "generation_time_s": round(elapsed, 1),
    }
    attack_results.append(result)

    status = "FOOLED!" if fooled else "CAUGHT"
    print(f"    Original:  Pred={pred_orig}, AI_prob={probs_orig[1]:.4f}")
    print(f"    Rewritten: Pred={pred_attack}, AI_prob={probs_attack[1]:.4f} [{status}]")

# =============================================
# 5. Summary & Analysis
# =============================================
print("\n" + "=" * 60)
print("ADVERSARIAL ATTACK SUMMARY")
print("=" * 60)

total = len(attack_results)
fooled_count = sum(1 for r in attack_results if r["fooled_detector"])
caught_count = total - fooled_count

print(f"  Total attack samples:   {total}")
print(f"  Fooled detector:        {fooled_count} ({fooled_count/total*100:.1f}%)" if total > 0 else "  No results")
print(f"  Caught by detector:     {caught_count} ({caught_count/total*100:.1f}%)" if total > 0 else "")

if attack_results:
    avg_orig_prob = np.mean([r["original_ai_prob"] for r in attack_results])
    avg_attack_prob = np.mean([r["attack_ai_prob"] for r in attack_results])
    print(f"\n  Avg AI prob (originals): {avg_orig_prob:.4f}")
    print(f"  Avg AI prob (attacks):   {avg_attack_prob:.4f}")
    print(f"  Avg prob shift:          {avg_attack_prob - avg_orig_prob:+.4f}")

# =============================================
# 6. Show detailed examples
# =============================================
print("\n" + "=" * 60)
print("DETAILED EXAMPLES")
print("=" * 60)

for i, r in enumerate(attack_results[:3]):  # Show first 3
    print(f"\n--- Example {i+1} ---")
    print(f"  Prompt style: {r['prompt_style']}")
    print(f"  Original (first 150 chars): {r['original_snippet'][:150]}...")
    print(f"  Rewritten (first 150 chars): {r['generated_snippet'][:150]}...")
    print(f"  Original AI prob:  {r['original_ai_prob']:.4f} → Pred: {r['original_pred']}")
    print(f"  Rewritten AI prob: {r['attack_ai_prob']:.4f} → Pred: {r['attack_pred']}")
    print(f"  {'=> FOOLED the detector!' if r['fooled_detector'] else '=> Detector caught it.'}")

# =============================================
# 7. Save all results
# =============================================
output_data = {
    "detector": DETECTOR_PATH,
    "generator": GEN_MODEL_NAME,
    "num_samples": total,
    "fooled_count": fooled_count,
    "caught_count": caught_count,
    "fooled_rate": round(fooled_count / total * 100, 1) if total > 0 else 0,
    "attack_results": attack_results,
}

with open(os.path.join(OUTPUT_DIR, "adversarial_results.json"), 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"\nResults saved to {OUTPUT_DIR}/adversarial_results.json")
print("Part 3 complete!")