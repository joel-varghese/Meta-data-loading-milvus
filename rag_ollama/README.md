# RAG with Ollama

Ollama-only stack: embeddings and LLM both via Ollama API. Simplest local RAG.

- **Data**: Milvus FAQ markdown (`milvus_docs/en/faq/*.md`)
- **Embeddings**: nomic-embed-text (768d) via Ollama
- **Vector DB**: Milvus server (localhost:19530)
- **LLM**: llama3.2:3b via LiteLLM
