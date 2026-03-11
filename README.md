# Milvus + Ollama RAG Applications

A collection of RAG (Retrieval-Augmented Generation) implementations using Milvus as the vector database. Each variant demonstrates different ingestion, retrieval, and LLM integration patterns.

## Project Structure

```
Milvus_ollama_trial/
├── shared/                 # Shared utilities (embeddings, context assembly)
├── data/                   # Shared data (CSV, etc.)
├── milvus_docs/            # Milvus FAQ/docs (for some RAGs)
├── rag_docling/            # Docling + hierarchical chunking
├── rag_graph/              # Graph-based RAG (entities, relations, passages)
├── rag_metadata/           # Metadata filtering (IMDB)
├── rag_ollama/             # Ollama-only stack (embeddings + LLM)
├── evaluate/               # RAG evaluation (RAGAS, etc.)
├── pyproject.toml
└── README.md
```

## RAG Implementations

| Folder | Description | Data | Key Features |
|--------|-------------|------|--------------|
| **rag_docling** | Document parsing with Docling | Milvus overview (URL) | HierarchicalChunker, Milvus Lite, local Llama |
| **rag_graph** | Knowledge-graph RAG | Job postings CSV | Entities, relations, passages; Milvus Cloud |
| **rag_metadata** | Metadata-filtered retrieval | IMDB CSV | Genre, rating, year filters; `shared` utilities |
| **rag_ollama** | Fully local via Ollama | Milvus FAQ markdown | nomic-embed-text, llama3.2 via LiteLLM |

## Adding a New RAG

1. Create a new folder, e.g. `rag_<name>/`
2. Add your notebooks and scripts there
3. Use project-root–aware paths in your code:

   ```python
   from pathlib import Path
   ROOT = Path.cwd() if (Path.cwd() / "shared").exists() else Path.cwd().parent
   # Use ROOT / "data" / "file.csv" etc.
   ```

4. Optionally import shared utilities:

   ```python
   import sys
   sys.path.insert(0, str(ROOT))
   from shared.milvus_utilities import embed_query, client_assemble_retrieved_context
   ```

5. Update this README with your new RAG’s description

## Setup

```bash
# Install dependencies
uv sync   # or: pip install -e .
```

## Data & Environment

- **data/** – Place `process.csv`, `all_job_post.csv`, etc. here
- **milvus_docs/** – Milvus FAQ markdown (for `rag_ollama`)
- **Environment variables**: `HF_TOKEN`, `MILVUS_API`, `MILVUS_CLOUD` (see notebooks)
