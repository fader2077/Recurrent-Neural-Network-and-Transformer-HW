import requests
import json

def query_ollama(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        return response.json()['response']
    except Exception as e:
        return f"Error calling Ollama: {e}"

def run_rag_pipeline(query):
    # 1. Retrieve & Rerank
    retrieved_docs = advanced_rag_retrieve(query, vector_db)
    
    # 2. Construct Prompt
    context_text = "\n\n".join([d.page_content for d in retrieved_docs])
    
    prompt = f"""
    <|start_header_id|>system<|end_header_id|>
    You are a helpful science assistant. Answer the question based ONLY on the context provided below.
    If the answer is not in the context, say "I don't know".
    
    Context:
    {context_text}
    <|eot_id|>
    <|start_header_id|>user<|end_header_id|>
    Question: {query}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
    
    # 3. Generate
    print("\nGenerating Answer...")
    answer = query_ollama(prompt)
    return answer

# --- Final Execution ---
q1 = "What is the equation for Newton's second law?"
answer1 = run_rag_pipeline(q1)
print(f"\nFinal Answer:\n{answer1}")

print("-" * 50)

q2 = "How do plants convert light?"
answer2 = run_rag_pipeline(q2)
print(f"\nFinal Answer:\n{answer2}")