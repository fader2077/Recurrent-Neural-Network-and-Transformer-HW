# HW3 Git Push Instructions

## Repository
- **URL:** https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git
- **Branch:** main
- **Folder:** hw3/

## Quick Push (Already Done)

The hw3 files have already been pushed to the repository. If you need to make changes and push again:

```bash
# 1. Navigate to the repo directory
cd D:\course\rnnlstm\Recurrent-Neural-Network-and-Transformer-HW

# 2. Check current status
git status

# 3. Stage changes (all hw3 files)
git add hw3/

# 4. Commit with message
git commit -m "Update HW3: RAG pipeline improvements"

# 5. Push to GitHub
git push origin main
```

## Clone from Scratch

If you need to set up on a new machine:

```bash
# 1. Clone the repository
git clone https://github.com/fader2077/Recurrent-Neural-Network-and-Transformer-HW.git

# 2. Navigate to hw3
cd Recurrent-Neural-Network-and-Transformer-HW/hw3

# 3. Install Python dependencies
pip install langchain langchain-community langchain-huggingface langchain-text-splitters chromadb sentence-transformers torch datasets matplotlib nbformat

# 4. Pull Ollama model (must have Ollama installed)
ollama pull llama3

# 5. Open and run hw3.ipynb in Jupyter or VS Code
```

## Files in hw3/

| File | Description |
|---|---|
| `hw3.ipynb` | Main notebook with all experiments |
| `report.md` | Markdown report |
| `report.tex` | LaTeX report source |
| `report.pdf` | Compiled PDF report |
| `advanced_experiment_summary.json` | Machine-readable summary of the corrected advanced pipeline |
| `advanced_rag_results.csv` | 50-question held-out advanced evaluation results |
| `Chunking.py` | Standalone chunking module |
| `DB.py` | Vector database module |
| `Retrieval.py` | Retrieval & re-ranking module |
| `Generation.py` | LLM generation module |
| `advanced_accuracy_comparison.png` | Advanced pipeline accuracy comparison figure |
| `chunk_size_distribution.png` | Chunk size comparison chart |
| `hit_rate_comparison.png` | Hit rate comparison chart |
| `latency_analysis.png` | Latency analysis chart |
| `evaluation_results.csv` | Full 50-question evaluation results |
| `experiment_summary.json` | Experiment metrics summary |
| `Homework 3.txt` | Original assignment specification |

## Recompile PDF Report

```bash
cd hw3
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex  # Run twice for ToC
```
