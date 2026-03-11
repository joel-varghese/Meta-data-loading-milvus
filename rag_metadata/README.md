# RAG with Metadata Filtering

IMDB metadata RAG with filterable fields (genre, rating, year). Uses shared utilities for embeddings and context assembly.

- **Data**: IMDB CSV (`data/process.csv`, produced from `final_data.csv` via `data_process.ipynb`)
- **Embeddings**: BGE-large-en-v1.5 via `shared.milvus_utilities`
- **Vector DB**: Milvus Lite (`milvus_demo.db`)
- **Filters**: RatingValue, Genres, MovieYear, etc. (no LLM step – retrieval + display)
