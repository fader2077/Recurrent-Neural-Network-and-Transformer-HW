# Terminal commands
# pip install langchain langchain-community langchain-huggingface chromadb sentence-transformers torch
# ollama pull llama3

import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# --- TA Note: Use a small synthetic corpus for testing logic ---
# In the real assignment, students will load the Wikipedia parquet file.
raw_text_corpus = [
    "The mitochondrion is a double-membrane-bound organelle found in most eukaryotic organisms. It is often called the powerhouse of the cell.",
    "Mitochondria generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy.",
    "Newton's laws of motion are three physical laws that, together, laid the foundation for classical mechanics.",
    "The first law states that an object remains at rest or in uniform motion unless acted upon by a force.",
    "The second law states that the rate of change of momentum of an object is directly proportional to the force applied, or F=ma.",
    "Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.",
    "Large Language Models (LLMs) are AI systems capable of understanding and generating human language."
]

# Convert to LangChain Documents
documents = [Document(page_content=text, metadata={"source": f"doc_{i}"}) for i, text in enumerate(raw_text_corpus)]

# --- Strategy A: Fixed-Size Chunking (Naive) ---
# Small chunks, strict cut-off
splitter_a = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
docs_a = splitter_a.split_documents(documents)
print(f"Strategy A (Fixed): Created {len(docs_a)} chunks.")

# --- Strategy B: Semantic/Larger Chunking ---
# Larger chunks to preserve context (Paragraph level)
splitter_b = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)
docs_b = splitter_b.split_documents(documents)
print(f"Strategy B (Semantic): Created {len(docs_b)} chunks.")