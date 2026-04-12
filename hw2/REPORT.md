# HW2: Detect AI Generated Text — Comprehensive Report

**Course**: RNN and Transformer  
**Assignment**: HW2 – Detect AI Generated Text  
**Hardware**: NVIDIA RTX 4090 GPU (24GB VRAM)  
**Dataset**: DAIGT V2 Train Dataset (44,868 essays; Label 0 = Human, Label 1 = AI)

---

## 1. Data Analysis & Baseline (Part 1)

### 1.1 Exploratory Data Analysis

We analyzed four text features comparing human-written (Label 0) and AI-generated (Label 1) essays:

| Feature | Human (mean) | AI (mean) | Cohen's d | p-value (Mann-Whitney U) |
|---------|-------------|-----------|-----------|--------------------------|
| Avg Word Length | 4.82 | 5.06 | −0.733 | ≈ 0 |
| Word Count | 271.3 | 280.5 | +0.594 | ≈ 0 |
| Sentence Count | 11.4 | 13.2 | +0.384 | ≈ 0 |
| Type-Token Ratio (TTR) | 0.807 | 0.841 | −0.366 | ≈ 0 |

All four features show **statistically significant differences** (p ≈ 0). The strongest signal is **average word length** (Cohen's d = −0.733, medium effect), indicating that AI text systematically uses longer words. Feature correlation analysis reveals word count and sentence count are highly correlated (r = 0.80), while TTR is negatively correlated with word count (r = −0.59).

**Key Insight**: These surface-level lexical patterns alone explain why even simple classifiers achieve >0.99 AUC — the AI text leaves a measurable distributional footprint.

### 1.2 Baseline Models (TF-IDF Features)

We implemented four classifiers using TF-IDF vectorization (max_features=10,000, ngram_range=(1,2)):

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 |
|-------|---------|----------|-----------|--------|-----|
| Logistic Regression | 0.9993 | 0.9939 | 0.9980 | 0.9863 | 0.9921 |
| **Linear SVM** | **0.9997** | **0.9970** | 0.9971 | 0.9951 | 0.9961 |
| Multinomial Naive Bayes | 0.9955 | 0.9692 | 0.9781 | 0.9423 | 0.9598 |
| Random Forest (200 trees) | 0.9987 | 0.9884 | 0.9968 | 0.9734 | 0.9850 |

**Finding**: Linear SVM achieves the highest baseline AUC (0.9997), approaching BERT-level performance. This confirms that the task contains strong lexical signals distinguishable by linear models. The TF-IDF + Logistic Regression baseline (AUC = 0.9993) serves as our benchmark that BERT models must beat.

### 1.3 Visualizations

- **EDA distributions** ([eda_distributions.png](results_baseline/eda_distributions.png)): 2×2 grid showing word count, TTR, sentence count, and average word length distributions for human vs AI text.
- **Enhanced EDA** ([enhanced_eda.png](results_baseline/enhanced_eda.png)): Feature correlation heatmap and all-baselines ROC comparison.
- **Confusion matrices** ([baseline_confusion_matrices.png](results_baseline/baseline_confusion_matrices.png)): Side-by-side confusion matrices for all four baselines.
- **ROC curve** ([baseline_roc_curve.png](results_baseline/baseline_roc_curve.png)): ROC curve for TF-IDF + LR baseline.

---

## 2. BERT Fine-Tuning & Scaling (Part 2)

### 2.1 Experimental Setup

- **Tokenizer**: `AutoTokenizer.from_pretrained('bert-base-cased')` / `'bert-large-cased'`
- **Training**: HuggingFace `Trainer` API with fp16 mixed precision
- **Data split**: 80/20 train/validation (stratified)
- **Optimizer**: AdamW, learning_rate = 2e-5, weight decay = 0.01, warmup ratio = 0.1

### 2.2 Results

| Experiment | Parameters | ROC-AUC | Accuracy | Batch Size | Training Time |
|------------|-----------|---------|----------|------------|---------------|
| BERT-base (3 epochs, max_len=512) | 108M | 0.9999 | 0.9941 | 32 | 8.3 min |
| BERT-base (3 epochs, max_len=256) | 108M | 0.9997 | 0.9899 | 32 | 4.0 min |
| BERT-base (5 epochs, max_len=512) | 108M | 0.9999 | 0.9967 | 32 | 14.3 min |
| BERT-large (3 epochs, max_len=512) | 334M | 0.9999 | 0.9974 | 16 | 26.1 min |

### 2.3 Loss Curves

Training and evaluation loss curves are shown in:
- [bert-base-cased_curves.png](results_bert-base-cased/bert-base-cased_curves.png)
- [bert-large-cased_curves.png](results_bert-large-cased/bert-large-cased_curves.png)
- [comparison_curves.png](results_bert-base-cased/comparison_curves.png): Side-by-side loss, ROC, and AUC bar chart
- [epoch_ablation.png](results_bert-base-cased/epoch_ablation.png): 3-epoch vs 5-epoch training loss comparison

### 2.4 Scaling Analysis

#### Model Size (Base vs Large)
- Both models achieve **identical ROC-AUC (0.9999)**.
- BERT-large slightly improves accuracy: 0.9974 vs 0.9941 (+0.33 percentage points).
- BERT-large has a **lower false positive rate** (0.26% vs 0.86%) — it makes fewer errors classifying human text as AI.
- However, BERT-large requires **3× parameters** (334M vs 108M) and **3.1× training time** (26.1 vs 8.3 min).

**Conclusion**: The marginal accuracy gain does not justify the computational cost. The task saturates at the base model capacity, indicating that the discriminative signal is well-captured by 108M parameters.

#### Sequence Length (512 vs 256)
- Reducing max_length from 512 to 256 causes only a **0.0002 AUC drop** (0.9999 → 0.9997) while **halving training time** (8.3 → 4.0 min).
- This suggests most classification signal is concentrated in the first 256 tokens.
- **Practical implication**: For deployment, 256-token models offer nearly identical performance at half the inference cost.

#### Epoch Count (3 vs 5)
- 5 epochs improves accuracy from 0.9941 to 0.9967 while maintaining the same AUC (0.9999).
- Loss curves confirm convergence by epoch 2–3; epochs 4–5 provide marginal gains without overfitting.
- **No overfitting observed** — eval loss remains stable through epoch 5.

### 2.5 Hypothesis: Does the Large model significantly outperform the Base model?

**No.** Both models achieve ROC-AUC = 0.9999 on this task. The task is "saturated" — the discriminative signal between human and AI text is so strong that even a 108M-parameter model captures it fully. The primary differences are in error profiles: BERT-large makes slightly fewer false-positive errors (misclassifying human text as AI), but the overall classification performance is identical. The task's saturation is further evidenced by the strong TF-IDF + SVM baseline (AUC = 0.9997).

### 2.6 Confidence Analysis

Confidence distribution analysis ([bert_confusion_confidence.png](results_bert-base-cased/bert_confusion_confidence.png)) shows:
- Both models are **extremely confident**: average P(AI) for human text is < 0.01, and average P(AI) for AI text is > 0.997.
- The bimodal confidence distribution indicates clear class separation with minimal uncertainty.

---

## 3. Adversarial Attack with Local LLM (Part 3)

### 3.1 Setup

- **Local LLM**: DeepSeek-R1:8B (deepseek-r1:8b-llama-distill-fp16) deployed via Ollama
- **Detector**: Best BERT model (BERT-large-cased, 3 epochs, max_len=512)
- **Essays selected**: 10 human-written essays from the validation set
- **Attack strategies**:
  - **Academic**: "Rewrite this essay to sound like it was written by a high school student for an English class."
  - **Stylistic**: "Rewrite this essay with a more casual, conversational tone while keeping the core arguments."
  - **Paraphrase**: "Completely paraphrase this essay using different vocabulary and sentence structures."

### 3.2 Single-Pass Attack Results (30 attacks = 10 essays × 3 strategies)

| Strategy | Attacks | Fooled | Fool Rate | Avg P(AI) After |
|----------|---------|--------|-----------|-----------------|
| Academic | 10 | 1 | 10% | ~0.91 |
| Stylistic | 10 | 0 | 0% | ~0.995 |
| Paraphrase | 10 | 2 | 20% | ~0.80 |
| **Total** | **30** | **3** | **10%** | — |

### 3.3 Iterative Attack Results (5 essays × up to 3 refinement rounds)

We implemented a feedback-loop attack: after each rewrite, we report the AI probability back to the LLM and ask it to further revise:

| Essay | Original P(AI) | Iter 1 P(AI) | Iter 2 P(AI) | Iter 3 P(AI) | Fooled? |
|-------|---------------|-------------|-------------|-------------|---------|
| #8927 | 0.00001 | 0.99999 | 0.99999 | 0.99999 | No |
| #7166 | 0.00002 | 0.99999 | 0.99999 | 0.99999 | No |
| #1744 | 0.00001 | 0.99999 | 0.99999 | 0.99999 | No |
| #1448 | 0.00001 | 0.99999 | 0.99999 | 0.99999 | No |
| #5258 | 0.00001 | 0.99999 | 0.99999 | 0.99999 | No |

**Result**: 0/5 fooled (0%). Iterative refinement did not improve the attack success rate.

### 3.4 Example Adversarial Attacks

#### Example 1: Essay #1448 — Paraphrase Strategy (FOOLED ✓)

**Original (human-written, P(AI) = 0.00001)**:
> "The author claim is that NASA is working on a way to visit venus despite how dangerous it is. They are planning on how they are going to visit Venus wihtout being harmed..."

**Rewritten (P(AI) = 0.2354 — below threshold, classified as "Human")**:
> "The author argues that NASA is developing a way to visit Venus despite its potential dangers. They aim to find a method to explore the planet safely, driven by the allure of discovering new things..."

**Why it fooled BERT**: The paraphrase smoothed grammatical errors and simplified vocabulary but preserved a student-like argumentative structure. The short length (reducing word count) also gave BERT less signal to work with.

#### Example 2: Essay #1448 — Academic Strategy (FOOLED ✓)

**Rewritten (P(AI) = 0.1159 — below threshold)**:
> "The essay you provided is already written in a style that mirrors how a student might present their thoughts, with some grammatical imperfections and conversational tone. However, to make it sound even more like a real student..."

**Why it fooled BERT**: The LLM included meta-commentary about the rewriting task, which created an unusual text pattern that didn't match BERT's learned features for either human or AI text.

#### Example 3: Essay #8927 — All Strategies (CAUGHT ✗)

**Original P(AI) = 0.00001 (human)**

| Strategy | P(AI) After | Result |
|----------|------------|--------|
| Academic | 0.99999 | Caught |
| Stylistic | 0.99999 | Caught |
| Paraphrase | 0.99999 | Caught |

**Why BERT succeeded**: All three rewrites introduced DeepSeek-R1's characteristic patterns (structured paragraphs, consistent grammar, topic sentences), which BERT correctly identified as AI-generated.

### 3.5 Feature Analysis of Attacks

| Feature Change | Caught (avg) | Fooled (avg) |
|---------------|-------------|-------------|
| Δ Word Count | −5 | −30 |
| Δ TTR | +0.13 | +0.23 |
| Δ Avg Word Length | +0.2 | +0.3 |
| Avg P(AI) | 0.9998 | 0.12 |

Successful attacks show **larger vocabulary diversity changes** (Δ TTR = +0.23 vs +0.13) and **greater word count reduction** — shorter text gives the detector less signal.

### 3.6 Why BERT is Robust

1. **Deep semantic understanding**: BERT captures paragraph flow, discourse structure, and vocabulary distribution patterns that survive single-pass rewrites.
2. **LLM-rewriting paradox**: Using an LLM to rewrite text *replaces one AI signature with another*. In the iterative attack, human essays became *more* AI-like after rewriting (P(AI) jumped from ~0.00001 to ~0.99999).
3. **Training diversity**: The DAIGT V2 dataset contains AI text from multiple LLMs, making the detector generalize well across generator styles.
4. **Feature redundancy**: Both lexical features (word length, TTR) and deep features (attention patterns) point to the same classification decision.

### 3.7 Potential Weaknesses

- Multi-turn human-in-the-loop editing (manual sentence-level refinement)
- Sentence-level mixing (alternating human and AI sentences within one essay)
- Training a specialized adversarial paraphraser optimized against the detector's gradient signals

---

## 4. Summary

| Model | ROC-AUC | Accuracy | Type |
|-------|---------|----------|------|
| TF-IDF + LR | 0.9993 | 0.9939 | Baseline |
| TF-IDF + SVM | 0.9997 | 0.9970 | Baseline |
| BERT-base (3ep) | 0.9999 | 0.9941 | Deep Learning |
| BERT-base (5ep) | 0.9999 | 0.9967 | Deep Learning |
| BERT-large (3ep) | 0.9999 | 0.9974 | Deep Learning |

**Core findings**:
1. Both BERT models beat the TF-IDF baseline, but overall performance differences are small, reflecting task saturation.
2. BERT-large offers no meaningful ROC-AUC improvement over BERT-base, despite 3× the parameters and training time.
3. The BERT detector is highly robust against adversarial rewriting: only 3/30 single-pass attacks succeeded, and 0/5 iterative attacks succeeded.
4. The "rewriting paradox" — LLM rewriting introduces new AI patterns — is the fundamental reason adversarial attacks fail.

---

## Environment

- **GPU**: NVIDIA GeForce RTX 4090 (24GB VRAM)
- **Python**: 3.13
- **PyTorch**: with CUDA + fp16 mixed precision
- **Transformers**: HuggingFace
- **Local LLM**: Ollama + DeepSeek-R1:8B

## Source Code & Repository

- **Notebook**: [HW2.ipynb](HW2.ipynb)
- **Scripts**: [Baseline.py](Baseline.py), [BERT.py](BERT.py), [LocalLLM.py](LocalLLM.py)
- **GitHub**: [https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW/tree/main/hw2](https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW/tree/main/hw2)
