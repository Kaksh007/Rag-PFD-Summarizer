# Deployment Guide

## Local Demo

Use Streamlit when you want the easiest portfolio walkthrough:

```powershell
streamlit run streamlit_app.py
```

Use FastAPI when you want to demonstrate backend integration:

```powershell
uvicorn src.api:app --reload
```

## Environment Variables

Required for all deployments:

- `VECTOR_DB`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIMENSION`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `LLM_TIMEOUT_SECONDS`

Required for local ChromaDB:

- `CHROMA_PATH`
- `COLLECTION_PREFIX`

Required for Pinecone:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_CLOUD`
- `PINECONE_REGION`
- `PINECONE_NAMESPACE`
- `PINECONE_METRIC`

Optional:

- `RERANK_ENABLED`
- `RERANK_MODEL`
- `CITATION_STYLE`

## Hosted Notes

- ChromaDB is best for local demos or hosts with durable disk.
- Pinecone is better for public demos because the vector index lives outside
  the web server.
- Ollama must be reachable from the deployed app. For public deployments, use a
  reachable Ollama-compatible endpoint or extend `src/rag.py` with another LLM
  provider.

## Docker

```powershell
docker build -t document-intelligence-rag .
docker run --env-file .env -p 8501:8501 document-intelligence-rag
```

For FastAPI in Docker, override the command:

```powershell
docker run --env-file .env -p 8000:8000 document-intelligence-rag `
  uvicorn src.api:app --host 0.0.0.0 --port 8000
```
