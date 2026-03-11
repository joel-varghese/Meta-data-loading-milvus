# RAG with Docling

Docling-based document parsing with hierarchical chunking. Uses Milvus Lite for storage and local Llama for generation.

- **Data**: Milvus overview (fetched from URL)
- **Embeddings**: BGE-large-en-v1.5 (SentenceTransformers)
- **Vector DB**: Milvus Lite (`milvus_demo.db`)
- **LLM**: Llama-3.2-1B-Instruct (local)
