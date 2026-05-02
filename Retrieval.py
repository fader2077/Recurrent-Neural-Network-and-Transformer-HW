from sentence_transformers import CrossEncoder
import torch

# --- Load Cross-Encoder (The Re-ranker) ---
# This model takes (Query, Document) pairs and outputs a similarity score.
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
print(f"Loading Cross-Encoder: {RERANK_MODEL_NAME} on CUDA...")

reranker = CrossEncoder(RERANK_MODEL_NAME, device='cuda')

def advanced_rag_retrieve(query, db, top_k_retrieval=5, top_k_rerank=3):
    """
    Stage 1: Vector Search (Fast, High Recall)
    Stage 2: Cross-Encoder Re-ranking (Slow, High Precision)
    """
    
    # --- Stage 1: Dense Retrieval ---
    print(f"\nQuery: {query}")
    initial_docs = db.similarity_search(query, k=top_k_retrieval)
    print(f"Stage 1: Retrieved {len(initial_docs)} candidates via Vector Search.")
    
    # --- Stage 2: Re-ranking ---
    # Prepare pairs for the Cross-Encoder: [[Query, Doc1], [Query, Doc2]...]
    doc_texts = [d.page_content for d in initial_docs]
    pairs = [[query, doc_text] for doc_text in doc_texts]
    
    # Predict scores (higher is better)
    scores = reranker.predict(pairs)
    
    # Combine docs with scores
    scored_docs = list(zip(initial_docs, scores))
    
    # Sort by score descending
    scored_docs_sorted = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    
    # Select Top-N
    final_docs = [doc for doc, score in scored_docs_sorted[:top_k_rerank]]
    
    # Debug Print: Show re-ranking effect
    print("Stage 2: Top Re-ranked Results:")
    for i, (doc, score) in enumerate(scored_docs_sorted[:top_k_rerank]):
        print(f"  [{i+1}] Score: {score:.4f} | Text: {doc.page_content[:50]}...")
        
    return final_docs

# Test the retrieval
test_query = "What generates energy in the cell?"
context_docs = advanced_rag_retrieve(test_query, vector_db)