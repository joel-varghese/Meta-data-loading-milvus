# Graph-based RAG

Knowledge-graph RAG with job postings. Extracts entities (job titles, skills, categories) and relations, stores in multiple Milvus collections.

- **Data**: Job postings CSV (`data/all_job_post.csv`)
- **Processing**: `kaggle_data.ipynb` – triple extraction; `graph_rag_milvus.ipynb` – full pipeline
- **Embeddings**: BGE-large-en-v1.5
- **Vector DB**: Milvus Cloud (requires `MILVUS_API`, `MILVUS_CLOUD`)
- **LLM**: Llama-3.2-1B-Instruct (local)
