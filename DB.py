from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- TA Note: Use a high-quality model compatible with RTX 4090 ---
EMBED_MODEL_NAME = "BAAI/bge-m3" # Or "sentence-transformers/all-MiniLM-L6-v2" for speed
print(f"Loading Embedding Model: {EMBED_MODEL_NAME} on CUDA...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL_NAME,
    model_kwargs={'device': 'cuda'}, # Utilizing RTX 4090
    encode_kwargs={'normalize_embeddings': True}
)

# Build Vector DB (Using Strategy B for this demo)
print("Building ChromaDB Index...")
vector_db = Chroma.from_documents(
    documents=docs_b,
    embedding=embeddings,
    collection_name="science_knowledge_base"
)

print("Vector DB ready.")