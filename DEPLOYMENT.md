# Deployment Guide

## Recommended Hosting

Deploy on Streamlit Community Cloud or any host that supports Python + Streamlit.

## Environment Variables

Set these in deployment secrets:

- `APP_MODE`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `TOP_K`
- `EMBEDDING_MODEL`
- `CHROMA_PATH`
- `COLLECTION_PREFIX`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`
- `LLM_TIMEOUT_SECONDS`

## Notes

- This project uses local Ollama by default for a free setup.
- For hosted/public demos, run against a reachable LLM endpoint compatible with the Ollama API shape (or extend `src/rag.py` provider logic).
- Persist `CHROMA_PATH` to durable storage if your host supports it.

## Public Demo Steps

1. Open app URL.
2. Upload a PDF.
3. Click **Index PDF**.
4. Click **Summarize document**.
5. Ask one or two questions and inspect retrieved context preview.
