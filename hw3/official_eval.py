import json
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from datasets import load_dataset
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
TRAIN_CSV = Path(r"D:\course\rnnlstm\HW3\kaggle-llm-science-exam\train.csv")
WIKI_CACHE = ROOT / "official_wiki_articles.json"
INDEX_A_DIR = ROOT / "faiss_index_a"
INDEX_B_DIR = ROOT / "faiss_index_b"
SUMMARY_JSON = ROOT / "official_experiment_summary.json"
RESULTS_CSV = ROOT / "official_evaluation_results.csv"
HIT_RATE_CSV = ROOT / "official_hit_rate_comparison.csv"
EXAMPLES_JSON = ROOT / "official_reranking_examples.json"
RERANK_CANDIDATES_CSV = ROOT / "reranking_example_candidates.csv"
RERANK_SELECTED_JSON = ROOT / "reranking_examples_selected.json"

EMBED_MODEL_NAME = "BAAI/bge-m3"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
TOP_K_INITIAL = 50
TOP_N_FINAL = 3

SCIENCE_KEYWORDS = [
    "physics", "chemistry", "biology", "cell", "atom", "molecule", "energy",
    "force", "gravity", "evolution", "photosynthesis", "DNA", "RNA", "protein",
    "electron", "neutron", "proton", "quantum", "thermodynamics", "entropy",
    "wave", "light", "electromagnetic", "radiation", "nuclear", "element",
    "periodic table", "chemical bond", "reaction", "acid", "base", "enzyme",
    "mitochondria", "chromosome", "gene", "mutation", "natural selection",
    "ecosystem", "planet", "solar system", "galaxy", "star", "telescope",
    "Newton", "Einstein", "relativity", "velocity", "acceleration", "momentum",
    "magnetism", "electric", "circuit", "voltage", "current", "resistance",
    "optics", "lens", "refraction", "diffraction", "frequency", "wavelength",
    "temperature", "pressure", "volume", "gas", "liquid", "solid",
    "organism", "species", "taxonomy", "bacteria", "virus", "immune",
    "neuron", "brain", "nervous system", "respiratory", "circulatory",
    "heart", "blood", "oxygen", "carbon", "nitrogen", "hydrogen",
    "climate", "atmosphere", "ocean", "continent", "mineral", "rock",
    "fossil", "plate tectonics", "volcano", "earthquake",
    "mathematics", "calculus", "algebra", "geometry", "probability",
    "computer science", "algorithm", "data structure",
]

STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "being", "by", "as", "at",
    "from", "that", "this", "these", "those", "it", "its", "their", "into",
    "which", "what", "who", "whom", "when", "where", "why", "how", "does",
    "did", "do", "about", "following", "statement", "statements", "most",
    "main", "purpose", "used", "using", "during", "after", "before",
}

GENERIC_ANSWER_TERMS = {
    "question", "answer", "choice", "choices", "following", "provided", "excerpt",
    "according", "wikipedia", "which", "what", "when", "where", "who", "whom",
    "whose", "main", "most", "term", "used", "using", "common", "known", "about",
    "statement", "statements", "describe", "described", "describes", "during",
    "after", "before", "based", "city", "country", "state", "film", "series",
    "episode", "people", "population", "area", "part", "current", "primary",
    "purpose", "feature", "features", "role", "significance",
    "has", "have", "many", "some", "made", "make", "works", "work", "served",
    "serves", "became", "become", "later", "through", "around",
}

SYSTEM_PROMPT = (
    "You are a precise science question-answering assistant. "
    "Use only the retrieved context. If the context is weak or conflicting, pick the "
    "best-supported option from A, B, C, D, E and do not invent facts. "
    "Output exactly one letter: A, B, C, D, or E."
)


def choose_device() -> str:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in the current Python environment. "
            "HW3 must run on GPU; install a CUDA-enabled PyTorch build first."
        )
    return "cuda"


def load_eval_df() -> pd.DataFrame:
    train_df = pd.read_csv(TRAIN_CSV)
    eval_df = train_df.sample(n=50, random_state=42)
    eval_df["source_row_index"] = eval_df.index
    eval_df = eval_df.reset_index(drop=True)
    eval_df = eval_df.rename(columns={"prompt": "question"})
    return eval_df


def load_or_collect_wiki_articles(max_articles: int = 500):
    if WIKI_CACHE.exists():
        # Tolerate UTF-8 BOM if the cache file was rewritten by external tools.
        return json.loads(WIKI_CACHE.read_text(encoding="utf-8-sig"))

    wiki_ds = load_dataset(
        "wikimedia/wikipedia",
        "20231101.simple",
        split="train",
        streaming=True,
    )
    collected_articles = []
    seen_titles = set()
    for article in wiki_ds:
        if len(collected_articles) >= max_articles:
            break
        title_lower = article["title"].lower()
        text_lower = article["text"][:500].lower()
        for keyword in SCIENCE_KEYWORDS:
            keyword_lower = keyword.lower()
            if keyword_lower in title_lower or keyword_lower in text_lower:
                if article["title"] not in seen_titles and len(article["text"]) > 200:
                    collected_articles.append(
                        {"title": article["title"], "text": article["text"]}
                    )
                    seen_titles.add(article["title"])
                break

    WIKI_CACHE.write_text(
        json.dumps(collected_articles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return collected_articles


def build_raw_documents(collected_articles):
    return [
        Document(
            page_content=article["text"],
            metadata={"title": article["title"], "source": "simple_wikipedia"},
        )
        for article in collected_articles
    ]


def make_chunkers():
    splitter_a = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    splitter_b = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter_a, splitter_b


def chunk_stats(docs):
    lengths = [len(doc.page_content) for doc in docs]
    return {
        "chunks": int(len(docs)),
        "avg_chars": float(np.mean(lengths)),
        "min_chars": int(np.min(lengths)),
        "max_chars": int(np.max(lengths)),
    }


def build_embeddings(device: str):
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL_NAME,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_or_build_faiss(index_dir: Path, docs, embeddings):
    if index_dir.exists():
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    vector_db = FAISS.from_documents(docs, embeddings)
    vector_db.save_local(str(index_dir))
    return vector_db


def tokenize(text: str):
    return [t for t in re.findall(r"[A-Za-z0-9]+", str(text).lower()) if len(t) > 2]


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def unique_terms(text: str, min_len: int = 4):
    seen = set()
    terms = []
    for token in re.findall(r"[a-z0-9]+", normalize_text(text)):
        if len(token) < min_len or token in STOPWORDS or token in GENERIC_ANSWER_TERMS:
            continue
        if token not in seen:
            terms.append(token)
            seen.add(token)
    return terms


def answer_key_terms(question: str, answer_text: str):
    question_terms = set(unique_terms(question))
    answer_terms = unique_terms(answer_text, min_len=3)
    preferred = [term for term in answer_terms if term not in question_terms]
    return preferred or answer_terms


def snippet_match_features(snippet: str, question: str, answer_text: str):
    snippet_norm = normalize_text(snippet)
    question_terms = unique_terms(question)
    answer_terms = answer_key_terms(question, answer_text)
    answer_norm = normalize_text(answer_text)

    exact_answer_match = len(answer_norm) >= 8 and answer_norm in snippet_norm
    answer_hits = [term for term in answer_terms if term in snippet_norm]
    question_hits = [term for term in question_terms if term in snippet_norm]

    strict_answer_bearing = exact_answer_match or (
        len(answer_hits) >= 2 and len(question_hits) >= 2
    )
    return {
        "exact_answer_match": exact_answer_match,
        "answer_hits": answer_hits,
        "question_hits": question_hits,
        "strict_answer_bearing": strict_answer_bearing,
    }


def is_strict_official_example(
    vector_top1_meta: dict,
    reranked_top1_meta: dict,
    reranked_top1_original_rank: int | None,
    vector_top1_score: float | None,
    reranked_top1_score: float | None,
) -> bool:
    return bool(
        not vector_top1_meta["strict_answer_bearing"]
        and reranked_top1_meta["strict_answer_bearing"]
        and reranked_top1_original_rank is not None
        and reranked_top1_original_rank > 1
        and vector_top1_score is not None
        and reranked_top1_score is not None
        and reranked_top1_score > vector_top1_score
    )


def is_acceptable_official_example(
    vector_top1_meta: dict,
    reranked_top1_meta: dict,
    reranked_top1_original_rank: int | None,
) -> bool:
    vector_is_weak = (
        not vector_top1_meta["strict_answer_bearing"]
        and len(vector_top1_meta["question_hits"]) <= 1
    )
    reranked_is_answer_bearing = reranked_top1_meta["strict_answer_bearing"]
    reranked_has_enough_evidence = (
        len(reranked_top1_meta["answer_hits"]) >= 2
        and len(reranked_top1_meta["question_hits"]) >= 2
    )
    moved_up = reranked_top1_original_rank is not None and reranked_top1_original_rank > 1
    return bool(vector_is_weak and moved_up and (reranked_is_answer_bearing or reranked_has_enough_evidence))


def is_advanced_branch_example(
    vector_top1_meta: dict,
    reranked_top1_meta: dict,
    reranked_top1_original_rank: int | None,
    vector_top1_score: float | None,
    reranked_top1_score: float | None,
) -> bool:
    vector_is_weaker = (
        not vector_top1_meta["strict_answer_bearing"]
        or len(vector_top1_meta["question_hits"]) <= len(reranked_top1_meta["question_hits"])
    )
    reranked_is_answer_bearing = reranked_top1_meta["strict_answer_bearing"]
    moved_up = reranked_top1_original_rank is not None and reranked_top1_original_rank > 1
    score_improved = (
        vector_top1_score is not None
        and reranked_top1_score is not None
        and reranked_top1_score > vector_top1_score
    )
    return bool(vector_is_weaker and reranked_is_answer_bearing and moved_up and score_improved)


def keyword_hit_score(chunk_text: str, answer_text: str) -> int:
    chunk_tokens = set(tokenize(chunk_text))
    answer_tokens = [t for t in tokenize(answer_text) if t not in STOPWORDS]
    return sum(token in chunk_tokens for token in answer_tokens)


def vector_search_only(query: str, vector_db, k: int = TOP_N_FINAL):
    return vector_db.similarity_search(query, k=k)


def vector_search_with_reranking(
    query: str,
    vector_db,
    reranker,
    initial_k: int = TOP_K_INITIAL,
    final_k: int = TOP_N_FINAL,
):
    candidates = vector_db.similarity_search(query, k=initial_k)
    if not candidates:
        return [], [], []
    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
    scored = sorted(zip(scores, candidates, range(1, len(candidates) + 1)), key=lambda x: x[0], reverse=True)
    return [doc for score, doc, rank in scored[:final_k]], scored, candidates


def compute_hit_rate(eval_df: pd.DataFrame, vector_db, reranker):
    vector_hits = 0
    rerank_hits = 0
    for _, row in eval_df.iterrows():
        answer_text = row[row["answer"]]
        vector_docs = vector_search_only(row["question"], vector_db, k=TOP_N_FINAL)
        reranked_docs, _, _ = vector_search_with_reranking(
            row["question"], vector_db, reranker
        )
        vector_hit = any(
            keyword_hit_score(doc.page_content, answer_text) > 0 for doc in vector_docs
        )
        rerank_hit = any(
            keyword_hit_score(doc.page_content, answer_text) > 0 for doc in reranked_docs
        )
        vector_hits += int(vector_hit)
        rerank_hits += int(rerank_hit)

    total = len(eval_df)
    return {
        "vector_only": vector_hits / total,
        "vector_rerank": rerank_hits / total,
    }


def overlap_terms(question: str, answer_text: str, chunk_text: str):
    focus_terms = [
        t for t in tokenize(f"{question} {answer_text}") if t not in STOPWORDS
    ]
    chunk_tokens = set(tokenize(chunk_text))
    return [term for term in focus_terms if term in chunk_tokens]


def choose_reranking_examples(eval_df: pd.DataFrame, vector_db, reranker, count: int = 2):
    ranked_examples = []
    for _, row in eval_df.iterrows():
        reranked_docs, scored, candidates = vector_search_with_reranking(
            row["question"], vector_db, reranker
        )
        if not reranked_docs or not candidates:
            continue
        vector_top = candidates[0]
        rerank_top = reranked_docs[0]
        if vector_top.page_content == rerank_top.page_content:
            continue

        vector_score = None
        rerank_score = None
        vector_rank_after = None
        for rank_after, (score, doc, original_rank) in enumerate(scored, start=1):
            if doc.page_content == vector_top.page_content and vector_score is None:
                vector_score = float(score)
                vector_rank_after = rank_after
            if doc.page_content == rerank_top.page_content and rerank_score is None:
                rerank_score = float(score)

        answer_text = row[row["answer"]]
        vector_overlap = overlap_terms(row["question"], answer_text, vector_top.page_content)
        rerank_overlap = overlap_terms(row["question"], answer_text, rerank_top.page_content)
        overlap_gain = len(rerank_overlap) - len(vector_overlap)
        if overlap_gain < 0:
            continue

        promoted_terms = ", ".join(rerank_overlap[:8]) or "none"
        demoted_terms = ", ".join(vector_overlap[:8]) or "none"
        explanation = (
            "The reranker promoted the new top chunk because it aligned better with the "
            f"question and correct answer terms. Promoted chunk matches: {promoted_terms}. "
            f"Original vector top-1 matches: {demoted_terms}."
        )
        ranked_examples.append(
            {
                "overlap_gain": overlap_gain,
                "question": row["question"],
                "correct_answer": row["answer"],
                "correct_answer_text": answer_text,
                "vector_top1_chunk": vector_top.page_content[:700],
                "vector_top1_rerank_score": vector_score,
                "vector_top1_rerank_rank": vector_rank_after,
                "reranked_top1_chunk": rerank_top.page_content[:700],
                "reranked_top1_score": rerank_score,
                "reranked_overlap_terms": rerank_overlap[:12],
                "vector_overlap_terms": vector_overlap[:12],
                "explanation": explanation,
            }
        )

    ranked_examples.sort(
        key=lambda item: (
            item["overlap_gain"],
            (item["reranked_top1_score"] or -9999.0) - (item["vector_top1_rerank_score"] or -9999.0),
            len(item["reranked_overlap_terms"]),
        ),
        reverse=True,
    )
    filtered = [
        item for item in ranked_examples
        if item["overlap_gain"] > 0 and len(item["reranked_overlap_terms"]) >= 2
    ]
    chosen = filtered[:count] if len(filtered) >= count else ranked_examples[:count]
    for item in chosen:
        item.pop("overlap_gain", None)
    return chosen


def search_reranking_example_candidates(eval_df: pd.DataFrame, vector_db, reranker):
    rows = []
    for eval_idx, row in eval_df.iterrows():
        question = row["question"]
        answer_text = row[row["answer"]]
        reranked_docs, scored, candidates = vector_search_with_reranking(
            question, vector_db, reranker
        )
        if not reranked_docs or not candidates:
            continue

        vector_top1 = candidates[0]
        reranked_top1 = reranked_docs[0]

        vector_meta = snippet_match_features(vector_top1.page_content, question, answer_text)
        rerank_meta = snippet_match_features(reranked_top1.page_content, question, answer_text)

        vector_score = None
        reranked_top1_score = None
        reranked_top1_original_rank = None
        top3_answer_bearing_rank_after = None
        top3_answer_bearing_original_rank = None
        top3_answer_bearing_score = None
        top3_answer_bearing_snippet = ""

        for rank_after, (score, doc, original_rank) in enumerate(scored, start=1):
            if doc.page_content == vector_top1.page_content and vector_score is None:
                vector_score = float(score)
            if doc.page_content == reranked_top1.page_content and reranked_top1_score is None:
                reranked_top1_score = float(score)
                reranked_top1_original_rank = int(original_rank)
            if rank_after <= 3 and top3_answer_bearing_rank_after is None:
                doc_meta = snippet_match_features(doc.page_content, question, answer_text)
                if doc_meta["strict_answer_bearing"]:
                    top3_answer_bearing_rank_after = rank_after
                    top3_answer_bearing_original_rank = int(original_rank)
                    top3_answer_bearing_score = float(score)
                    top3_answer_bearing_snippet = doc.page_content[:260].replace("\n", " ")

        question_overlap_gain = (
            len(rerank_meta["question_hits"]) - len(vector_meta["question_hits"])
        )
        answer_overlap_gain = len(rerank_meta["answer_hits"]) - len(vector_meta["answer_hits"])

        tier = ""
        reason = ""
        if is_strict_official_example(
            vector_meta,
            rerank_meta,
            reranked_top1_original_rank,
            vector_score,
            reranked_top1_score,
        ):
            tier = "Tier 1"
            reason = "vector top-1 is not answer-bearing, while reranked top-1 becomes answer-bearing and scores higher"
        elif (
            not vector_meta["strict_answer_bearing"]
            and top3_answer_bearing_rank_after is not None
            and top3_answer_bearing_original_rank is not None
            and top3_answer_bearing_original_rank > top3_answer_bearing_rank_after
        ):
            tier = "Tier 2"
            reason = (
                f"an answer-bearing chunk moved from vector rank {top3_answer_bearing_original_rank} "
                f"to rerank rank {top3_answer_bearing_rank_after}"
            )
        elif (
            reranked_top1.page_content != vector_top1.page_content
            and question_overlap_gain > 0
            and reranked_top1_original_rank is not None
            and reranked_top1_original_rank > 1
        ):
            tier = "Tier 3"
            reason = (
                f"reranked top-1 is more semantically aligned with the question "
                f"({len(rerank_meta['question_hits'])} question-term hits vs "
                f"{len(vector_meta['question_hits'])})"
            )
        else:
            continue

        rows.append(
            {
                "tier": tier,
                "eval_idx": int(eval_idx),
                "question": question,
                "correct_letter": row["answer"],
                "correct_answer_text": answer_text,
                "vector_top1_rank": 1,
                "vector_top1_score_if_available": vector_score,
                "vector_top1_contains_answer": bool(vector_meta["strict_answer_bearing"]),
                "vector_top1_question_hits": ", ".join(vector_meta["question_hits"][:8]),
                "vector_top1_answer_hits": ", ".join(vector_meta["answer_hits"][:8]),
                "vector_top1_snippet": vector_top1.page_content[:260].replace("\n", " "),
                "reranked_top1_original_rank": reranked_top1_original_rank,
                "reranked_top1_score": reranked_top1_score,
                "reranked_top1_contains_answer": bool(rerank_meta["strict_answer_bearing"]),
                "reranked_top1_question_hits": ", ".join(rerank_meta["question_hits"][:8]),
                "reranked_top1_answer_hits": ", ".join(rerank_meta["answer_hits"][:8]),
                "reranked_top1_snippet": reranked_top1.page_content[:260].replace("\n", " "),
                "reranked_top3_contains_answer": bool(top3_answer_bearing_rank_after is not None),
                "reranked_top3_best_rank": top3_answer_bearing_rank_after,
                "reranked_top3_best_original_rank": top3_answer_bearing_original_rank,
                "reranked_top3_best_score": top3_answer_bearing_score,
                "reranked_top3_best_snippet": top3_answer_bearing_snippet,
                "question_overlap_gain": int(question_overlap_gain),
                "answer_overlap_gain": int(answer_overlap_gain),
                "reason_for_candidate_selection": reason,
            }
        )

    candidates_df = pd.DataFrame(rows)
    if len(candidates_df) == 0:
        return candidates_df

    tier_order = {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}
    candidates_df["tier_priority"] = candidates_df["tier"].map(tier_order)
    candidates_df["selection_score"] = (
        (3 - candidates_df["tier_priority"]) * 1000
        + candidates_df["question_overlap_gain"] * 100
        + candidates_df["reranked_top1_original_rank"].fillna(1) * 10
        + candidates_df["answer_overlap_gain"] * 5
    )
    candidates_df = candidates_df.sort_values(
        ["tier_priority", "selection_score", "eval_idx"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return candidates_df.drop(columns=["tier_priority"])


def strict_official_reranking_candidates(eval_df: pd.DataFrame, vector_db, reranker):
    candidates_df = search_reranking_example_candidates(eval_df, vector_db, reranker)
    if len(candidates_df) == 0:
        return candidates_df
    return candidates_df[candidates_df["tier"] == "Tier 1"].reset_index(drop=True)


def acceptable_official_reranking_candidates(eval_df: pd.DataFrame, vector_db, reranker):
    candidates_df = search_reranking_example_candidates(eval_df, vector_db, reranker)
    if len(candidates_df) == 0:
        return candidates_df
    mask = []
    for _, row in candidates_df.iterrows():
        vector_meta = {
            "strict_answer_bearing": bool(row["vector_top1_contains_answer"]),
            "question_hits": [x for x in str(row["vector_top1_question_hits"]).split(", ") if x and x != "nan"],
            "answer_hits": [x for x in str(row["vector_top1_answer_hits"]).split(", ") if x and x != "nan"],
        }
        rerank_meta = {
            "strict_answer_bearing": bool(row["reranked_top1_contains_answer"]),
            "question_hits": [x for x in str(row["reranked_top1_question_hits"]).split(", ") if x and x != "nan"],
            "answer_hits": [x for x in str(row["reranked_top1_answer_hits"]).split(", ") if x and x != "nan"],
        }
        mask.append(
            is_acceptable_official_example(
                vector_meta,
                rerank_meta,
                int(row["reranked_top1_original_rank"]) if pd.notna(row["reranked_top1_original_rank"]) else None,
            )
        )
    return candidates_df[pd.Series(mask, index=candidates_df.index)].reset_index(drop=True)


def select_final_reranking_examples(candidates_df: pd.DataFrame, count: int = 2):
    if len(candidates_df) == 0:
        return [], "Warning: no re-ranking candidates were found."

    examples = []
    warning = ""
    grouped = {
        "Tier 1": candidates_df[candidates_df["tier"] == "Tier 1"],
        "Tier 2": candidates_df[candidates_df["tier"] == "Tier 2"],
        "Tier 3": candidates_df[candidates_df["tier"] == "Tier 3"],
    }
    for tier_name in ["Tier 1", "Tier 2", "Tier 3"]:
        for _, row in grouped[tier_name].iterrows():
            examples.append(row.to_dict())
            if len(examples) >= count:
                break
        if len(examples) >= count:
            break

    if len(grouped["Tier 1"]) + len(grouped["Tier 2"]) < count:
        warning = (
            "Warning: strict Tier 1 / Tier 2 answer-bearing examples were insufficient on the official "
            "50-question sample, so Tier 3 semantic-improvement examples were selected."
        )

    return examples[:count], warning


def build_mcq_prompt(question: str, row: pd.Series, top_docs):
    context_text = "\n\n".join(
        f"[Chunk {i + 1}]\n{doc.page_content[:900]}" for i, doc in enumerate(top_docs[:TOP_N_FINAL])
    )
    choices_text = "\n".join(
        f"{label}. {row[label]}" for label in ["A", "B", "C", "D", "E"]
    )
    return (
        f"{SYSTEM_PROMPT}\n"
        "Answer with exactly one uppercase letter and nothing else.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n"
        f"Choices:\n{choices_text}\n\n"
        "Output format:\nA\n\n"
        "Answer:"
    )


def call_ollama(prompt_text: str):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt_text,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 4, "stop": ["\n", " "]},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=180)
    response.raise_for_status()
    return response.json().get("response", "").strip()


def extract_answer_letter(response_text: str):
    match = re.search(r"\b([A-E])\b", response_text.strip().upper())
    return match.group(1) if match else "X"


def run_generation_eval(eval_df: pd.DataFrame, vector_db, reranker):
    rows = []
    vector_times = []
    rerank_times = []
    generation_times = []

    for idx, row in eval_df.iterrows():
        question = row["question"]

        t0 = time.time()
        candidates = vector_db.similarity_search(question, k=TOP_K_INITIAL)
        vector_times.append(time.time() - t0)

        t1 = time.time()
        pairs = [[question, doc.page_content] for doc in candidates]
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, doc in scored[:TOP_N_FINAL]]
        rerank_times.append(time.time() - t1)

        prompt_text = build_mcq_prompt(question, row, top_docs)
        t2 = time.time()
        response_text = call_ollama(prompt_text)
        generation_times.append(time.time() - t2)

        pred = extract_answer_letter(response_text)
        rows.append(
            {
                "question_idx": idx,
                "question": question,
                "correct_answer": row["answer"],
                "predicted_answer": pred,
                "is_correct": bool(pred == row["answer"]),
                "response": response_text,
            }
        )

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_CSV, index=False)

    return {
        "accuracy": float(results_df["is_correct"].mean()),
        "results_df": results_df,
        "avg_vector_search_time": float(np.mean(vector_times)),
        "avg_rerank_time": float(np.mean(rerank_times)),
        "avg_generation_time": float(np.mean(generation_times)),
        "avg_total_time": float(
            np.mean(np.array(vector_times) + np.array(rerank_times) + np.array(generation_times))
        ),
    }


def main():
    device = choose_device()
    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    eval_df = load_eval_df()
    articles = load_or_collect_wiki_articles(max_articles=500)
    raw_documents = build_raw_documents(articles)
    splitter_a, splitter_b = make_chunkers()
    docs_a = splitter_a.split_documents(raw_documents)
    docs_b = splitter_b.split_documents(raw_documents)

    embeddings = build_embeddings(device)
    print("Loading/building FAISS index A...")
    vector_db_a = load_or_build_faiss(INDEX_A_DIR, docs_a, embeddings)
    print("Loading/building FAISS index B...")
    vector_db_b = load_or_build_faiss(INDEX_B_DIR, docs_b, embeddings)
    print("Loading CrossEncoder reranker...")
    reranker = CrossEncoder(RERANK_MODEL_NAME, device=device)

    print("Computing hit rate comparison...")
    hit_rate_a = compute_hit_rate(eval_df, vector_db_a, reranker)
    hit_rate_b = compute_hit_rate(eval_df, vector_db_b, reranker)
    hit_rate_df = pd.DataFrame(
        [
            {"index": "Index A (fixed 500/50)", "method": "Vector Search Only", "hit_rate": hit_rate_a["vector_only"]},
            {"index": "Index A (fixed 500/50)", "method": "Vector Search + Re-ranking", "hit_rate": hit_rate_a["vector_rerank"]},
            {"index": "Index B (recursive 1000/200)", "method": "Vector Search Only", "hit_rate": hit_rate_b["vector_only"]},
            {"index": "Index B (recursive 1000/200)", "method": "Vector Search + Re-ranking", "hit_rate": hit_rate_b["vector_rerank"]},
        ]
    )
    hit_rate_df.to_csv(HIT_RATE_CSV, index=False)

    print("Selecting qualitative reranking examples...")
    examples = choose_reranking_examples(eval_df, vector_db_b, reranker, count=2)
    EXAMPLES_JSON.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Running Ollama generation evaluation on official eval_df...")
    generation_metrics = run_generation_eval(eval_df, vector_db_a, reranker)

    summary = {
        "dataset": {
            "source_file": str(TRAIN_CSV),
            "train_rows": int(pd.read_csv(TRAIN_CSV).shape[0]),
            "eval_rows": int(len(eval_df)),
            "sampling": 'train_df = pd.read_csv(train_path); official_eval_df = train_df.sample(n=50, random_state=42).reset_index(drop=True); official_eval_df = official_eval_df.rename(columns={"prompt": "question"})',
            "sampled_source_row_indices": [int(x) for x in eval_df["source_row_index"].tolist()],
        },
        "device": device,
        "corpus": {
            "raw_documents": int(len(raw_documents)),
            "raw_total_chars": int(sum(len(doc.page_content) for doc in raw_documents)),
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
            "official_index": "A",
            "official_accuracy": float(generation_metrics["accuracy"]),
        },
        "latency": {
            "avg_vector_search_time": generation_metrics["avg_vector_search_time"],
            "avg_rerank_time": generation_metrics["avg_rerank_time"],
            "avg_generation_time": generation_metrics["avg_generation_time"],
            "avg_total_time": generation_metrics["avg_total_time"],
        },
        "reranking_examples_file": EXAMPLES_JSON.name,
        "appendix_reference": "advanced_experiment_summary.json",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
