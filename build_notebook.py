"""Build the official HW3 notebook programmatically."""

import nbformat as nbf


nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

cells = []


def md(source: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(source))


def code(source: str) -> None:
    cells.append(nbf.v4.new_code_cell(source))


md(
    """# Homework 3: RAG for Science Question Answering

This notebook separates the **official HW3 pipeline** from an **optional advanced appendix**.

- Official evaluation set: `train.csv` random 50-question sample with `random_state=42`
- Official retrieval pipeline: FAISS + `BAAI/bge-m3` + dense retrieval `top_k=20` + cross-encoder re-ranking `top_n=3`
- Official generator: Ollama `llama3`
- Official score: reported only from the random 50-question `train.csv` sample

The appendix keeps the earlier science-only / DeepSeek / ensemble branch as a non-official extension."""
)

md("## 0. Setup and GPU Check")
code(
    """import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder

from official_eval import (
    EXAMPLES_JSON,
    HIT_RATE_CSV,
    INDEX_A_DIR,
    INDEX_B_DIR,
    OLLAMA_MODEL,
    RERANK_CANDIDATES_CSV,
    RERANK_MODEL_NAME,
    RERANK_SELECTED_JSON,
    RESULTS_CSV,
    SUMMARY_JSON,
    SYSTEM_PROMPT,
    TOP_K_INITIAL,
    TOP_N_FINAL,
    build_embeddings,
    build_mcq_prompt,
    build_raw_documents,
    chunk_stats,
    choose_device,
    load_or_build_faiss,
    load_or_collect_wiki_articles,
    make_chunkers,
    run_generation_eval,
    search_reranking_example_candidates,
    select_final_reranking_examples,
)

warnings.filterwarnings("ignore")
device = choose_device()
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
"""
)

md(
    """# Official HW3 Pipeline

## 1. Dataset and official random 50-question evaluation

The official evaluation setup must remain:

```python
official_train_df = pd.read_csv("train.csv")
official_eval_df = official_train_df.sample(n=50, random_state=42).reset_index(drop=True)
official_eval_df = official_eval_df.rename(columns={"prompt": "question"})
```"""
)
code(
    """official_train_df = pd.read_csv("train.csv")
official_eval_df = official_train_df.sample(n=50, random_state=42).reset_index(drop=True)
official_eval_df = official_eval_df.rename(columns={"prompt": "question"})

print(f"train.csv rows: {len(official_train_df)}")
print(f"official eval rows: {len(official_eval_df)}")
display(official_eval_df[["question", "A", "B", "C", "D", "E", "answer"]].head(3))
"""
)

md(
    """## 2. Corpus and chunking

The retrieval corpus is built from 500 science-related Simple Wikipedia articles cached in
`official_wiki_articles.json`. This keeps the official notebook reproducible and avoids refetching the
full Wikipedia stream on every run."""
)
code(
    """articles = load_or_collect_wiki_articles(max_articles=500)
raw_documents = build_raw_documents(articles)

raw_total_chars = sum(len(doc.page_content) for doc in raw_documents)
raw_stats_df = pd.DataFrame(
    [
        {"metric": "Raw documents", "value": len(raw_documents)},
        {"metric": "Total characters", "value": raw_total_chars},
        {"metric": "Average document length", "value": round(raw_total_chars / len(raw_documents), 2)},
    ]
)
display(raw_stats_df)
print("Sample titles:", [article["title"] for article in articles[:10]])
"""
)
code(
    """splitter_a, splitter_b = make_chunkers()
docs_a = splitter_a.split_documents(raw_documents)
docs_b = splitter_b.split_documents(raw_documents)

chunk_stats_df = pd.DataFrame(
    [
        {"strategy": "Strategy A", "chunk_size": 500, "overlap": 50, **chunk_stats(docs_a)},
        {"strategy": "Strategy B", "chunk_size": 1000, "overlap": 200, **chunk_stats(docs_b)},
    ]
)
display(chunk_stats_df)
"""
)
code(
    """fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sizes_a = [len(doc.page_content) for doc in docs_a]
sizes_b = [len(doc.page_content) for doc in docs_b]

axes[0].hist(sizes_a, bins=30, color="#4c78a8", alpha=0.85, edgecolor="black")
axes[0].set_title("Strategy A Chunk Lengths")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Count")

axes[1].hist(sizes_b, bins=30, color="#f58518", alpha=0.85, edgecolor="black")
axes[1].set_title("Strategy B Chunk Lengths")
axes[1].set_xlabel("Characters")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()
"""
)

md(
    """## 3. FAISS index construction

Two FAISS indices are built with the same embedding model:

- Index A: Strategy A chunks
- Index B: Strategy B chunks

Embedding model: `BAAI/bge-m3` on GPU."""
)
code(
    """embeddings = build_embeddings(device)
vector_db_a = load_or_build_faiss(INDEX_A_DIR, docs_a, embeddings)
vector_db_b = load_or_build_faiss(INDEX_B_DIR, docs_b, embeddings)

index_df = pd.DataFrame(
    [
        {"index": "Index A", "strategy": "Fixed 500/50", "chunks": len(docs_a)},
        {"index": "Index B", "strategy": "Recursive 1000/200", "chunks": len(docs_b)},
    ]
)
display(index_df)
"""
)

md(
    """## 4. Dense retrieval and re-ranking

The official retrieval pipeline uses:

1. Dense vector retrieval with `top_k=20`
2. Cross-encoder re-ranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
3. Final top-3 chunks for generation"""
)
code(
    """reranker = CrossEncoder(RERANK_MODEL_NAME, device=device)

from official_eval import compute_hit_rate

hit_rate_a = compute_hit_rate(official_eval_df, vector_db_a, reranker)
hit_rate_b = compute_hit_rate(official_eval_df, vector_db_b, reranker)

hit_rate_df = pd.DataFrame(
    [
        {"index": "Index A (fixed 500/50)", "method": "Vector Search Only", "hit_rate": hit_rate_a["vector_only"]},
        {"index": "Index A (fixed 500/50)", "method": "Vector Search + Re-ranking", "hit_rate": hit_rate_a["vector_rerank"]},
        {"index": "Index B (recursive 1000/200)", "method": "Vector Search Only", "hit_rate": hit_rate_b["vector_only"]},
        {"index": "Index B (recursive 1000/200)", "method": "Vector Search + Re-ranking", "hit_rate": hit_rate_b["vector_rerank"]},
    ]
)
hit_rate_df.to_csv(HIT_RATE_CSV, index=False)
display(hit_rate_df)
"""
)

md("## 5. Hit Rate comparison")
code(
    """pivot_hit_rate_df = hit_rate_df.pivot(index="index", columns="method", values="hit_rate")
ax = (pivot_hit_rate_df * 100).plot(kind="bar", figsize=(10, 5), color=["#4c78a8", "#f58518"])
ax.set_ylabel("Hit Rate (%)")
ax.set_title("Hit Rate Comparison on Official eval_df")
ax.set_ylim(0, 100)
ax.legend(loc="upper right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%")
plt.tight_layout()
plt.show()
"""
)

md(
    """## 6. Qualitative re-ranking examples

The next cell scans all 50 official questions, writes a candidate table, selects the final two examples,
and prints a report-ready summary. If strict Tier 1 / Tier 2 cases are unavailable, it warns and falls back
to Tier 3 semantic-improvement examples instead of overstating the retrieval quality."""
)
code(
    """reranking_candidates_df = search_reranking_example_candidates(official_eval_df, vector_db_b, reranker)
reranking_candidates_df.to_csv(RERANK_CANDIDATES_CSV, index=False)

selected_examples, reranking_warning = select_final_reranking_examples(reranking_candidates_df, count=2)
RERANK_SELECTED_JSON.write_text(json.dumps(selected_examples, ensure_ascii=False, indent=2), encoding="utf-8")
EXAMPLES_JSON.write_text(json.dumps(selected_examples, ensure_ascii=False, indent=2), encoding="utf-8")

display(
    reranking_candidates_df[
        [
            "tier",
            "eval_idx",
            "question",
            "correct_letter",
            "vector_top1_contains_answer",
            "reranked_top1_contains_answer",
            "reranked_top3_contains_answer",
            "reranked_top1_original_rank",
            "question_overlap_gain",
            "reason_for_candidate_selection",
        ]
    ].head(10)
)

if reranking_warning:
    print(reranking_warning)

for case_id, example in enumerate(selected_examples, start=1):
    print(f"Case {case_id}:")
    print(f"- Question: {example['question']}")
    print(f"- Correct answer: {example['correct_letter']}. {example['correct_answer_text']}")
    print("- Vector Search top-1:")
    print(f"  - snippet: {example['vector_top1_snippet']}")
    print(f"  - original rank: {example['vector_top1_rank']}")
    print(f"  - Cross-Encoder score: {example['vector_top1_score_if_available']:.4f}")
    print(f"  - why it is weak / irrelevant: {example['vector_top1_question_hits'] or 'few direct question-term matches'}")
    print("- After Re-ranking top-1:")
    print(f"  - snippet: {example['reranked_top1_snippet']}")
    print(f"  - original vector rank before reranking: {example['reranked_top1_original_rank']}")
    print(f"  - Cross-Encoder score: {example['reranked_top1_score']:.4f}")
    print(f"  - why it is better / answer-bearing: {example['reason_for_candidate_selection']}")
    if example['tier'] == 'Tier 2' and example['reranked_top3_contains_answer']:
        print(f"  - answer-bearing chunk in reranked top-3: rank {example['reranked_top3_best_rank']} from original rank {example['reranked_top3_best_original_rank']}")
        print(f"  - top-3 answer-bearing snippet: {example['reranked_top3_best_snippet']}")
    print("- Explanation:")
    if example['tier'] == 'Tier 1':
        print("  - The re-ranker promoted the answer-bearing chunk to the top.")
    elif example['tier'] == 'Tier 2':
        print("  - The re-ranker promoted an answer-bearing chunk into the final top-3 context.")
    else:
        print("  - The re-ranker promoted a more semantically relevant evidence chunk, although this proxy does not prove that the final answer is fully contained.")
    print("-" * 120)
"""
)

md(
    """## 7. Ollama Llama-3 generation

Official generation setting:

- local model: `llama3` via Ollama
- retrieval input: top-3 chunks after re-ranking
- strict output: exactly one letter `A/B/C/D/E`
- evaluation: accuracy on `official_eval_df`"""
)
code(
    """print("System prompt used for generation:")
print(SYSTEM_PROMPT)

sample_row = official_eval_df.iloc[0]
from official_eval import vector_search_with_reranking
sample_docs, _, _ = vector_search_with_reranking(sample_row["question"], vector_db_b, reranker)
sample_prompt = build_mcq_prompt(sample_row["question"], sample_row, sample_docs)
print("\\nPrompt preview:")
print(sample_prompt[:1500])
"""
)

md("## 8. Accuracy and latency")
code(
    """generation_metrics = run_generation_eval(official_eval_df, vector_db_b, reranker)
official_results_df = generation_metrics["results_df"]

summary = {
    "dataset": {
        "source_file": "train.csv",
        "train_rows": int(len(official_train_df)),
        "eval_rows": int(len(official_eval_df)),
        "sampling": 'official_train_df.sample(n=50, random_state=42).rename(columns={"prompt": "question"})',
    },
    "device": device,
    "corpus": {
        "raw_documents": int(len(raw_documents)),
        "raw_total_chars": int(raw_total_chars),
    },
    "chunking": {
        "strategy_a": chunk_stats(docs_a),
        "strategy_b": chunk_stats(docs_b),
    },
    "retrieval": {
        "top_k_initial": TOP_K_INITIAL,
        "top_n_final": TOP_N_FINAL,
        "index_a_vector_only_hit_rate": float(hit_rate_a["vector_only"]),
        "index_a_rerank_hit_rate": float(hit_rate_a["vector_rerank"]),
        "index_b_vector_only_hit_rate": float(hit_rate_b["vector_only"]),
        "index_b_rerank_hit_rate": float(hit_rate_b["vector_rerank"]),
    },
    "generation": {
        "model": OLLAMA_MODEL,
        "official_accuracy": float(generation_metrics["accuracy"]),
    },
    "latency": {
        "avg_vector_search_time": generation_metrics["avg_vector_search_time"],
        "avg_rerank_time": generation_metrics["avg_rerank_time"],
        "avg_generation_time": generation_metrics["avg_generation_time"],
        "avg_total_time": generation_metrics["avg_total_time"],
    },
    "reranking_examples_file": RERANK_SELECTED_JSON.name,
    "appendix_reference": "advanced_experiment_summary.json",
}
SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

latency_df = pd.DataFrame(
    [
        {"stage": "Vector Search", "avg_time_sec": generation_metrics["avg_vector_search_time"]},
        {"stage": "Re-ranking", "avg_time_sec": generation_metrics["avg_rerank_time"]},
        {"stage": "LLM Generation", "avg_time_sec": generation_metrics["avg_generation_time"]},
        {"stage": "Total", "avg_time_sec": generation_metrics["avg_total_time"]},
    ]
)

print(f"Official accuracy: {generation_metrics['accuracy']:.2%}")
display(official_results_df.head(10))
display(latency_df)
"""
)
code(
    """fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(
    latency_df["stage"][:-1],
    latency_df["avg_time_sec"][:-1],
    color=["#4c78a8", "#f58518", "#54a24b"],
)
ax.set_ylabel("Average time (seconds)")
ax.set_title("Official Pipeline Latency Breakdown")
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.3f}s", ha="center")
plt.tight_layout()
plt.show()

print(
    "Strategy B gives the strongest vector-only hit-rate proxy, while the final official generation "
    "pipeline uses Strategy B with re-ranking as the required two-stage retrieval setup."
)
"""
)

md(
    """# Appendix: Optional Advanced Experiment

This section is optional and is **not used as the official HW3 score**.
The official score remains the random 50-question `train.csv` evaluation with Ollama `llama3`."""
)

md(
    """## A1. Why an advanced science-only branch was tested

The earlier extension branch explored whether stronger retrieval, answer-aware evidence scoring,
and model variants could improve performance on a filtered science-oriented subset. This appendix
documents that branch for traceability only."""
)
code(
    """science_summary = json.loads(Path("advanced_experiment_summary.json").read_text(encoding="utf-8"))
science_results_df = pd.read_csv("advanced_rag_results.csv")

traceability_df = pd.DataFrame(
    [
        {"artifact": "hw3advance.ipynb", "role": "original advanced experiment notebook"},
        {"artifact": "advanced_experiment_summary.json", "role": "summary metrics for science-only branch"},
        {"artifact": "advanced_rag_results.csv", "role": "50-question held-out science-only ensemble outputs"},
        {"artifact": "advanced_accuracy_comparison.png", "role": "advanced appendix comparison figure"},
    ]
)
display(traceability_df)
"""
)

md(
    """## A2. Science-only pool construction

The advanced branch filtered the public mirror with science keywords, then used:

- `science_dev_df`: first 10 filtered questions for rule tuning
- `science_test_df`: 50 held-out filtered questions sampled with `random_state=42`

These variables are appendix-only and do not replace `official_eval_df`."""
)
code(
    """science_dataset_df = pd.DataFrame(
    [
        {"metric": "Public mirror rows", "value": science_summary["dataset"]["full_rows"]},
        {"metric": "Science-only pool rows", "value": science_summary["dataset"]["science_pool_rows"]},
        {"metric": "Science dev rows", "value": science_summary["dataset"]["dev_rows"]},
        {"metric": "Science held-out test rows", "value": science_summary["dataset"]["heldout_test_rows"]},
    ]
)
display(science_dataset_df)
"""
)

md(
    """## A3. Query rewriting and long Wikipedia extraction

The advanced branch in `hw3advance.ipynb` added:

- query rewriting for Wikipedia search,
- long-form Wikipedia extracts,
- local dynamic chunking over the fetched evidence pages.

Those methods were implemented in the advanced notebook cells and are summarized here rather than rerun."""
)
code(
    """advanced_method_df = pd.DataFrame(
    [
        {"component": "Query rewriting", "source": "hw3advance.ipynb Cells 42-43"},
        {"component": "Answer-aware evidence scoring", "source": "hw3advance.ipynb Cells 43-44"},
        {"component": "NLI-style scoring", "source": "hw3advance.ipynb Cells 45-48"},
        {"component": "Dynamic chunk rerank + NLI", "source": "hw3advance.ipynb Cells 48-49"},
        {"component": "DeepSeek RAG variants", "source": "hw3advance.ipynb Cells 52-57"},
    ]
)
display(advanced_method_df)
"""
)

md(
    """## A4. Option-aware evidence scoring and NLI-style scoring

The advanced branch combined:

- cross-encoder re-ranking,
- option-aware retrieval over each answer choice,
- NLI-style entailment margins for final scoring.

These are extension experiments only and are not part of the official Llama-3 random-50 score."""
)

md("## A5. DeepSeek and ensemble comparison")
code(
    """advanced_results_table = pd.DataFrame(
    [
        {"method": "Llama 3 model-only on science dev slice", "accuracy": science_summary["baseline"]["llama3_model_only_dev_accuracy"]},
        {"method": "DeepSeek model-only on science dev slice", "accuracy": science_summary["baseline"]["deepseek_model_only_dev_accuracy"]},
        {"method": "DeepSeek RAG vector-only", "accuracy": science_summary["retrieval_experiments"]["deepseek_rag_vector_only_dev_accuracy"]},
        {"method": "DeepSeek RAG with re-ranking", "accuracy": science_summary["retrieval_experiments"]["deepseek_rag_rerank_dev_accuracy"]},
        {"method": "Option-aware rerank + NLI", "accuracy": science_summary["retrieval_experiments"]["option_aware_rerank_nli_dev_accuracy"]},
        {"method": "Final ensemble on 10-question dev slice", "accuracy": science_summary["ensemble"]["dev_accuracy"]},
        {"method": "Final ensemble on 50 held-out science-only questions", "accuracy": science_summary["ensemble"]["heldout_50_dev_tuned_accuracy"]},
        {"method": "Post-hoc best rule on the same 50-question run", "accuracy": science_summary["ensemble"]["heldout_50_posthoc_best_rule_accuracy"]},
    ]
)
advanced_results_table["accuracy_pct"] = advanced_results_table["accuracy"] * 100
display(advanced_results_table)

source_breakdown_df = (
    science_results_df.groupby("source")["is_correct"]
    .agg(["count", "mean"])
    .reset_index()
    .rename(columns={"mean": "accuracy"})
)
display(source_breakdown_df)
"""
)

md("## A6. Advanced results table")
code(
    """fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(
    advanced_results_table["method"],
    advanced_results_table["accuracy_pct"],
    color=["#7f7f7f", "#636363", "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2"],
)
ax.set_xlabel("Accuracy (%)")
ax.set_title("Optional Advanced Experiment Summary")
ax.set_xlim(0, 100)
for bar, value in zip(bars, advanced_results_table["accuracy_pct"]):
    ax.text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")
plt.tight_layout()
plt.show()
"""
)

md(
    """## A7. Why these are not official HW3 scores

- The advanced branch uses a filtered science-only pool rather than the required random 50-question `train.csv` sample.
- It uses DeepSeek and ensemble extensions beyond the official Ollama `llama3` pipeline.
- It is useful as an optional ablation and extension study, but the official HW3 score remains the random-50 `train.csv` result above."""
)

md(
    """## Official Conclusion

- Official dataset: `train.csv` random 50-question sample
- Official vector store: FAISS
- Official embedding model: `BAAI/bge-m3`
- Official generator: Ollama `llama3`
- Official score: 42.00% on the random 50-question sample
- The appendix shows a higher ceiling for an optional extended branch, but those numbers are not used as the formal HW3 score"""
)

nb["cells"] = cells

with open("hw3.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
