import json
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

import official_eval as oe


ROOT = Path(__file__).resolve().parent
END2END_CSV = ROOT / "end_to_end_ablation_results.csv"
END2END_JSON = ROOT / "end_to_end_ablation_summary.json"
TOPK_CSV = ROOT / "topk_sensitivity_results.csv"
TOPK_JSON = ROOT / "topk_sensitivity_summary.json"
QUERY_MODE_CSV = ROOT / "query_mode_comparison.csv"
QUERY_MODE_JSON = ROOT / "query_mode_comparison_summary.json"
ERROR_CSV = ROOT / "official_error_analysis.csv"
ERROR_JSON = ROOT / "official_error_analysis_summary.json"


def build_query(row: pd.Series, mode: str) -> str:
    if mode == "question_only":
        return row["question"]
    choices = "\n".join([f"{c}. {row[c]}" for c in ["A", "B", "C", "D", "E"]])
    return f"{row['question']}\nChoices:\n{choices}"


def tokenize(text: str):
    return [t for t in re.findall(r"[A-Za-z0-9]+", str(text).lower()) if len(t) > 2]


def contains_answer_terms(text: str, answer_text: str):
    chunk_tokens = set(tokenize(text))
    answer_tokens = [t for t in tokenize(answer_text) if t not in oe.STOPWORDS]
    hits = [t for t in answer_tokens if t in chunk_tokens]
    return bool(hits), hits


def retrieve_docs(question: str, db, reranker, top_k: int, rerank: bool):
    t0 = time.time()
    candidates = db.similarity_search(question, k=top_k if rerank else 3)
    vector_t = time.time() - t0
    if not rerank:
        return candidates[:3], vector_t, 0.0
    t1 = time.time()
    pairs = [[question, d.page_content] for d in candidates]
    scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
    ranked = sorted(zip(scores, candidates, range(1, len(candidates) + 1)), key=lambda x: x[0], reverse=True)
    top_docs = [d for _, d, _ in ranked[:3]]
    rerank_t = time.time() - t1
    return top_docs, vector_t, rerank_t


def run_generation(eval_df, db, reranker, top_k: int, rerank: bool):
    rows = []
    vec_ts, rr_ts, gen_ts = [], [], []
    for i, row in eval_df.iterrows():
        q = row["question"]
        top_docs, vt, rt = retrieve_docs(q, db, reranker, top_k=top_k, rerank=rerank)
        vec_ts.append(vt)
        rr_ts.append(rt)
        prompt = oe.build_mcq_prompt(q, row, top_docs)
        t2 = time.time()
        resp = oe.call_ollama(prompt)
        gt = time.time() - t2
        gen_ts.append(gt)
        pred = oe.extract_answer_letter(resp)
        rows.append(
            {
                "question_idx": i,
                "question": q,
                "correct_answer": row["answer"],
                "predicted_answer": pred,
                "is_correct": bool(pred == row["answer"]),
                "response": resp,
                "top_docs": [d.page_content[:500] for d in top_docs],
            }
        )
    df = pd.DataFrame(rows)
    return {
        "results": df,
        "accuracy": float(df["is_correct"].mean()),
        "correct": int(df["is_correct"].sum()),
        "avg_vector_latency": float(np.mean(vec_ts)),
        "avg_rerank_latency": float(np.mean(rr_ts)),
        "avg_llm_latency": float(np.mean(gen_ts)),
        "avg_total_latency": float(np.mean(np.array(vec_ts) + np.array(rr_ts) + np.array(gen_ts))),
    }


def hit_rate(eval_df, db, reranker, query_mode: str, top_k: int):
    vector_hits = 0
    rerank_hits = 0
    vec_t, rr_t = [], []
    for _, row in eval_df.iterrows():
        q = build_query(row, query_mode)
        answer_text = row[row["answer"]]
        t0 = time.time()
        cands = db.similarity_search(q, k=top_k)
        vec_t.append(time.time() - t0)
        vdocs = cands[:3]
        if any(oe.keyword_hit_score(d.page_content, answer_text) > 0 for d in vdocs):
            vector_hits += 1
        t1 = time.time()
        pairs = [[q, d.page_content] for d in cands]
        scores = reranker.predict(pairs, batch_size=32, show_progress_bar=False)
        ranked = sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)
        rdocs = [d for _, d in ranked[:3]]
        rr_t.append(time.time() - t1)
        if any(oe.keyword_hit_score(d.page_content, answer_text) > 0 for d in rdocs):
            rerank_hits += 1
    n = len(eval_df)
    return {
        "vector_hit_rate": vector_hits / n,
        "reranked_hit_rate": rerank_hits / n,
        "avg_vector_latency": float(np.mean(vec_t)),
        "avg_rerank_latency": float(np.mean(rr_t)),
    }


def main():
    eval_df = oe.load_eval_df()
    device = oe.choose_device()
    embeddings = oe.build_embeddings(device)
    db_a = FAISS.load_local(str(oe.INDEX_A_DIR), embeddings, allow_dangerous_deserialization=True)
    db_b = FAISS.load_local(str(oe.INDEX_B_DIR), embeddings, allow_dangerous_deserialization=True)
    reranker = CrossEncoder(oe.RERANK_MODEL_NAME, device=device)

    # Task 2: End-to-end ablation
    configs = [
        ("Index A + Vector-only + Llama3", db_a, False),
        ("Index A + Re-ranking + Llama3", db_a, True),
        ("Index B + Vector-only + Llama3", db_b, False),
        ("Index B + Re-ranking + Llama3", db_b, True),
    ]
    ablation_rows = []
    saved_runs = {}
    for name, db, use_rerank in configs:
        run = run_generation(eval_df, db, reranker, top_k=50, rerank=use_rerank)
        saved_runs[name] = run
        ablation_rows.append(
            {
                "Pipeline": name,
                "Retrieval setting": "dense top-3" if not use_rerank else "dense top_k=50 + rerank top-3",
                "Correct / 50": f"{run['correct']} / 50",
                "Accuracy": run["accuracy"],
                "Avg total latency": run["avg_total_latency"],
                "Avg vector latency": run["avg_vector_latency"],
                "Avg rerank latency": run["avg_rerank_latency"],
                "Avg LLM latency": run["avg_llm_latency"],
            }
        )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(END2END_CSV, index=False)
    END2END_JSON.write_text(
        json.dumps({"rows": ablation_rows, "official_pipeline": "Index B + Re-ranking + Llama3"}, indent=2),
        encoding="utf-8",
    )

    # Task 3: top-k sensitivity on Index B
    topk_rows = []
    for k in [20, 50]:
        hr = hit_rate(eval_df, db_b, reranker, query_mode="question_only", top_k=k)
        run = run_generation(eval_df, db_b, reranker, top_k=k, rerank=True)
        topk_rows.append(
            {
                "top_k": k,
                "Rerank top_n": 3,
                "Hit Rate": hr["reranked_hit_rate"],
                "Accuracy": run["accuracy"],
                "Vector latency": run["avg_vector_latency"],
                "Rerank latency": run["avg_rerank_latency"],
                "LLM latency": run["avg_llm_latency"],
                "Total latency": run["avg_total_latency"],
                "Vector-only Hit Rate": hr["vector_hit_rate"],
            }
        )
    topk_df = pd.DataFrame(topk_rows)
    topk_df.to_csv(TOPK_CSV, index=False)
    TOPK_JSON.write_text(json.dumps({"rows": topk_rows}, indent=2), encoding="utf-8")

    # Task 4: query mode comparison
    qrows = []
    for idx_name, db in [("A", db_a), ("B", db_b)]:
        for mode in ["question_only", "question_plus_choices"]:
            hr = hit_rate(eval_df, db, reranker, query_mode=mode, top_k=50)
            qrows.append(
                {
                    "Index": idx_name,
                    "Query mode": mode,
                    "Vector-only Hit Rate": hr["vector_hit_rate"],
                    "Reranked Hit Rate": hr["reranked_hit_rate"],
                    "Improvement pp": (hr["reranked_hit_rate"] - hr["vector_hit_rate"]) * 100,
                }
            )
    qdf = pd.DataFrame(qrows)
    qdf.to_csv(QUERY_MODE_CSV, index=False)
    QUERY_MODE_JSON.write_text(json.dumps({"rows": qrows}, indent=2), encoding="utf-8")

    # Task 5: error analysis from official pipeline run
    official_run = saved_runs["Index B + Re-ranking + Llama3"]
    wrong = official_run["results"][~official_run["results"]["is_correct"]].copy()
    error_rows = []
    for _, r in wrong.iterrows():
        src = eval_df.iloc[int(r["question_idx"])]
        ans_text = src[src["answer"]]
        snippets = r["top_docs"]
        joined = " ".join(snippets)
        has_ans, hits = contains_answer_terms(joined, ans_text)
        if not has_ans:
            cat = "Missing evidence"
        elif r["predicted_answer"] != src["answer"] and len(hits) <= 2:
            cat = "Partial / weak evidence"
        elif r["predicted_answer"] != src["answer"] and len(hits) > 2:
            cat = "LLM reasoning error"
        else:
            cat = "Distractor evidence"
        error_rows.append(
            {
                "question_idx": int(r["question_idx"]),
                "question": r["question"],
                "correct_answer": src["answer"],
                "predicted_answer": r["predicted_answer"],
                "top1_snippet": snippets[0],
                "top2_snippet": snippets[1],
                "top3_snippet": snippets[2],
                "contains_correct_answer_terms": has_ans,
                "answer_term_hits": ", ".join(hits[:10]),
                "error_category": cat,
            }
        )
    err_df = pd.DataFrame(error_rows)
    err_df.to_csv(ERROR_CSV, index=False)
    summary = Counter(err_df["error_category"]) if len(err_df) else {}
    ERROR_JSON.write_text(
        json.dumps(
            {
                "wrong_count": int(len(err_df)),
                "category_counts": dict(summary),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # figures
    plt.figure(figsize=(7, 4.8))
    x = np.arange(len(ablation_df))
    vals = ablation_df["Accuracy"].values * 100
    plt.bar(x, vals, color=["#7aa6ff", "#4f7ef7", "#89d9b2", "#2fb67f"])
    plt.xticks(x, ["A-Vec", "A-RR", "B-Vec", "B-RR"])
    plt.ylabel("Accuracy (%)")
    plt.title("End-to-end Ablation on Official Random-50")
    for i, v in enumerate(vals):
        plt.text(i, v + 0.6, f"{v:.1f}%", ha="center", fontsize=9)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(ROOT / "end_to_end_ablation.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    x = np.arange(len(topk_df))
    acc = topk_df["Accuracy"].values * 100
    hr = topk_df["Hit Rate"].values * 100
    w = 0.35
    plt.bar(x - w / 2, hr, width=w, label="Reranked Hit Rate")
    plt.bar(x + w / 2, acc, width=w, label="End-to-end Accuracy")
    plt.xticks(x, [f"top_k={k}" for k in topk_df["top_k"]])
    plt.ylim(0, 100)
    plt.title("Top-k Sensitivity (Index B, rerank top-3)")
    plt.ylabel("Percentage (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "topk_sensitivity.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.8))
    qplot = qdf.copy()
    qplot["label"] = qplot["Index"] + "-" + qplot["Query mode"].str.replace("question_", "q_")
    x = np.arange(len(qplot))
    v = qplot["Vector-only Hit Rate"].values * 100
    r = qplot["Reranked Hit Rate"].values * 100
    w = 0.35
    plt.bar(x - w / 2, v, width=w, label="Vector-only")
    plt.bar(x + w / 2, r, width=w, label="Reranked")
    plt.xticks(x, qplot["label"], rotation=20)
    plt.ylim(0, 100)
    plt.ylabel("Hit Rate (%)")
    plt.title("Query Mode Comparison (top_k=50)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT / "query_mode_comparison.png", dpi=200)
    plt.close()

    if len(err_df):
        cdf = err_df["error_category"].value_counts()
        plt.figure(figsize=(7, 4.8))
        plt.bar(cdf.index, cdf.values, color="#f29e4c")
        plt.title("Error Analysis Breakdown (Official Pipeline)")
        plt.ylabel("Count")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(ROOT / "error_analysis_breakdown.png", dpi=200)
        plt.close()

    # refresh latency figure from official summary
    summary_json = json.loads((ROOT / "official_experiment_summary.json").read_text(encoding="utf-8"))
    lat = summary_json["latency"]
    labels = ["Vector", "Re-rank", "LLM", "Total"]
    values = [
        lat["avg_vector_search_time"],
        lat["avg_rerank_time"],
        lat["avg_generation_time"],
        lat["avg_total_time"],
    ]
    plt.figure(figsize=(7, 4.8))
    plt.bar(labels, values, color=["#6aaed6", "#74c476", "#fd8d3c", "#9e9ac8"])
    plt.ylabel("Seconds")
    plt.title("Latency Analysis (Official Pipeline)")
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(ROOT / "latency_analysis.png", dpi=200)
    plt.close()

    print("done")


if __name__ == "__main__":
    main()
