# Homework 3: RAG for Science Question Answering

**Course:** RNN and Transformer  
**Date:** Spring 2026

---

## 1. Executive Summary

The original HW3 notebook produced two suspicious outcomes:

1. **Re-ranking showed no gain**.
2. **Final accuracy stayed at 20%**, exactly the random baseline for 5-choice multiple choice.

After re-running the assignment rigorously, the root causes were clear:

- the sampled 50-question set was not restricted to the science-like portion of the public dataset mirror,
- the initial 500-article Simple Wikipedia subset was too small and too noisy,
- the original hit-rate proxy was too weak to expose what the re-ranker was actually doing.

I therefore added a second, corrected experiment track:

- a **science-only evaluation pool**,
- **query rewriting** for Wikipedia retrieval,
- **long-form Wikipedia extracts**,
- **chunk-level re-ranking**,
- **option-aware evidence scoring**,
- **DeepSeek + Ollama** as an extended local-model comparison,
- a final **ensemble** over retrieval-heavy and reasoning-heavy components.

### Final Results

| Result | Accuracy |
|---|---:|
| Original Llama 3 baseline (broken setup) | 20.00% |
| Llama 3 model-only on corrected science dev slice | 30.00% |
| DeepSeek model-only on corrected science dev slice | 40.00% |
| DeepSeek RAG, vector-only retrieval (dev) | 30.00% |
| DeepSeek RAG, **with re-ranking** (dev) | **60.00%** |
| Final ensemble on 10-question dev slice | **80.00%** |
| Final ensemble on **50 held-out science questions** | **72.00%** |
| Post-hoc best rule on the completed 50-question run | **74.00%** |

**Important:** the system reached **80% on the development slice**, but the stricter 50-question held-out evaluation stabilized at **72%**, with **74%** as the best post-hoc rule over the completed run. I therefore cannot honestly claim a held-out 80% final result.

---

## 2. Dataset Correction

### 2.1 Original Public Mirror Problem

The public Hugging Face mirror used in the first notebook run contains many rows that are not cleanly science-focused. The original notebook sampled 50 questions uniformly from all 6,684 rows, which produced a mixed-domain evaluation slice and severely distorted the assignment.

### 2.2 Corrected Science-Only Pool

To fix this, I filtered the full dataset with a science keyword gate applied to the prompt and choices.

| Metric | Value |
|---|---:|
| Full public mirror size | 6,684 |
| Science-like pool after filtering | 1,469 |
| Dev slice | 10 |
| Held-out test slice | 50 |

The final evaluation therefore uses a **held-out 50-question science-only set**, not the original mixed 50-question sample.

---

## 3. Part 1: Indexing Pipeline

### 3.1 Original Chunking Experiment

The assignment requirement to compare chunking strategies was kept and re-used as the baseline experiment.

| Metric | Strategy A (Fixed) | Strategy B (Semantic/Recursive) |
|---|---:|---:|
| Chunk size | 500 chars | 1000 chars |
| Overlap | 50 chars | 200 chars |
| Number of chunks | 6,865 | 3,204 |
| Average chunk size | 334 chars | 733 chars |

**Observation:** Strategy B preserved paragraph-level coherence better than Strategy A, which reduced sentence fragmentation.

### 3.2 Why The Original Corpus Failed

The initial corpus was built from only **500 Simple Wikipedia pages** selected by broad keyword matching. That caused two problems:

- many pages were only loosely related to the final questions,
- even good re-ranking could not help much when the candidate pool was already weak.

### 3.3 Corrected Retrieval Corpus

The improved pipeline does not rely only on that static 500-page subset. Instead, it builds a **targeted Wikipedia evidence set** per question by:

1. rewriting the question into a concise search query,
2. fetching top Wikipedia pages with the MediaWiki API,
3. downloading longer extracts,
4. chunking the resulting pages,
5. applying dense retrieval + re-ranking over those chunks.

This correction is what unlocked the later re-ranking gains.

---

## 4. Part 2: Retrieval And Re-ranking

### 4.1 Why The Old Hit Rate Was Misleading

The first report used a coarse keyword-overlap hit rate. That metric only checked whether any keyword from the correct answer appeared somewhere in the retrieved top-3 chunks. It could not measure whether the **best chunk moved upward after re-ranking**.

That is why the first notebook showed:

| Configuration | Vector Only | + Re-ranking |
|---|---:|---:|
| Index A (Fixed Chunks) | 46.00% | 46.00% |
| Index B (Semantic Chunks) | 50.00% | 50.00% |

Those numbers were real for that weak metric, but they were not sufficient to judge the value of the re-ranker.

### 4.2 Corrected Retrieval Experiment

On the corrected science dev slice, I evaluated a stronger downstream metric: **final multiple-choice answer accuracy** after retrieval.

| Method | Dev Accuracy (10 questions) |
|---|---:|
| DeepSeek model-only | 40.00% |
| DeepSeek RAG, vector-only retrieval | 30.00% |
| DeepSeek RAG, **with re-ranking** | **60.00%** |
| Option-aware rerank + NLI | **60.00%** |

This is the key corrected finding for HW3:

> **Re-ranking doubled answer accuracy from 30% to 60% on the corrected science dev slice.**

That directly addresses the original concern that rerank should be better. Under a stronger corpus and a stronger evaluation metric, it is better.

### 4.3 Re-ranking Examples

Two representative cases from the corrected run:

1. **MOND / galaxy clusters**  
   The query-rewritten retrieval found the correct concept page for *Modified Newtonian dynamics*. Vector-only retrieval still mixed in less useful context, while the re-ranker promoted the chunk describing the discrepancy reduction from roughly 10 to roughly 2, enabling the correct answer.

2. **Roche limit**  
   The question-only retrieval found the correct page, but the best answer required the most precise chunk. After re-ranking, the chunk explicitly describing the tidal-disruption distance was moved to the top and the final system selected the correct option.

---

## 5. Part 3: Generation With Ollama

### 5.1 Required Ollama Integration

The notebook still implements the required Ollama generation step with **Llama 3**.

However, because the corrected science slice is much harder than the original broken sample, I also ran an extended comparison with **DeepSeek-R1 8B distill** already available in the local environment.

This extended comparison is clearly separated from the required Llama 3 baseline.

### 5.2 Model Comparison

| Model / Pipeline | Accuracy |
|---|---:|
| Llama 3 model-only (science dev) | 30.00% |
| DeepSeek model-only (science dev) | 40.00% |
| DeepSeek RAG + rerank (science dev) | 60.00% |
| Final ensemble (science dev) | 80.00% |

### 5.3 Final Ensemble

The strongest system combined three signals:

1. **Option-aware evidence scorer**  
   Uses query rewriting, long Wikipedia extracts, chunk-level re-ranking, and NLI-style scoring.

2. **Question-context DeepSeek RAG**  
   Retrieves Wikipedia context for the question, re-ranks the chunks, then answers with DeepSeek.

3. **DeepSeek model-only vote**  
   Used only when the faster evidence scorer is uncertain and the first two systems disagree.

Final decision rule:

- trust the option-aware scorer when its confidence gap is large,
- otherwise compare it with the question-context RAG answer,
- if needed, add the model-only vote,
- if all three disagree, ask a final DeepSeek judge over retrieved context.

### 5.4 Final Held-Out Result

| Evaluation | Accuracy |
|---|---:|
| Dev slice (10 science questions) | 80.00% |
| Held-out test (50 science questions, dev-tuned rule) | 72.00% |
| Held-out test (post-hoc best threshold over completed run) | 74.00% |

This is a large improvement over the original 20% baseline, but it still falls short of a stable held-out 80% result.

![Advanced Accuracy Comparison](advanced_accuracy_comparison.png)

---

## 6. Latency Discussion

The original static-vector baseline remained very fast:

| Stage | Average Time |
|---|---:|
| Vector search | 0.0218 s |
| Re-ranking | 0.0218 s |
| LLM generation (Llama 3) | 2.4894 s |

In the corrected advanced pipeline, the dominant new cost comes from:

- Wikipedia query rewriting and page fetch,
- chunk-level re-ranking over longer extracts,
- DeepSeek generation on harder questions.

So the advanced system is substantially slower, but the accuracy gain is also substantial:

- **20% -> 72% held-out**,
- **30% -> 60% on the corrected DeepSeek RAG ablation** when re-ranking is enabled.

This makes the engineering tradeoff reasonable: higher latency buys a much stronger answer quality signal.

---

## 7. What Was Fixed Relative To The First Submission

1. The evaluation set is now **science-only**, not a mixed random sample.
2. The corpus is now **targeted to the question domain**, not just a weak 500-page Simple Wikipedia slice.
3. Re-ranking is evaluated with a stronger downstream metric, and it now shows a clear gain.
4. The notebook now contains an **advanced experiment section** documenting the corrected pipeline.
5. The final report and PDF now reflect the corrected results.

---

## 8. Final Conclusion

The original notebook result was not trustworthy because the data slice and the corpus were both misaligned with the assignment goal. After correcting those issues, the homework now shows the behavior expected from a real RAG system:

- chunking matters,
- corpus quality matters even more,
- re-ranking becomes valuable when the candidate pool is relevant,
- local LLM choice changes the ceiling,
- ensemble-style evidence aggregation can materially improve accuracy.

The strongest corrected pipeline reached:

- **80% on a 10-question science dev slice**,
- **72% on a 50-question held-out science test**,
- **74% as the best post-hoc rule over the completed 50-question run**.

That is a substantial improvement from the initial 20% baseline and provides a much more defensible HW3 submission.
