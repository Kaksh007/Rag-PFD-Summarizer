# Document Intelligence RAG Platform

A portfolio-grade PDF question-answering and summarization system built around
retrieval-augmented generation. The app supports local ChromaDB indexing by
default and can switch to Pinecone for managed vector search.

## Why This Project Matters

This is more than a PDF summarizer. It demonstrates the architecture used in
real RAG systems:

- PDF ingestion with page-level metadata.
- Chunking with overlap for semantic retrieval.
- Sentence-transformer embeddings.
- Pluggable vector-store adapters for ChromaDB and Pinecone.
- LangChain prompt templates and LangGraph-compatible orchestration.
- Grounded answer generation with citations.
- Optional cross-encoder reranking.
- Streamlit UI for demos.
- FastAPI service for production-style integration.
- Lightweight RAG evaluation harness.
- Docker-ready deployment.

## Architecture

```text
PDF Upload
  -> Text Extraction
  -> Chunking + Metadata
  -> Embeddings
  -> Vector Store Adapter
       -> ChromaDB local
       -> Pinecone managed
  -> Retrieval
  -> Optional Reranking
  -> LangGraph RAG Workflow
  -> Ollama LLM Generation
  -> Cited Answer + Retrieved Sources
```

## Project Structure

```text
streamlit_app.py          Streamlit demo UI
src/api.py                FastAPI service
src/config.py             Environment-driven configuration
src/pdf_ingest.py         PDF text extraction
src/chunking.py           Chunk creation with metadata
src/vector_stores.py      ChromaDB and Pinecone adapters
src/embeddings_store.py   Backward-compatible vector-store facade
src/rag.py                RAG service and Ollama generation
src/rag_workflow.py       LangGraph-compatible RAG workflow
src/evaluation.py         Retrieval/answer quality checks
evals/rag_eval.py         CLI evaluation runner
```

## Setup

1. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create `.env` from `.env.example`.

```powershell
Copy-Item .env.example .env
```

4. Install and run Ollama, then pull a local model.

```powershell
ollama pull llama3.2:1b
```

## Run The Streamlit App

```powershell
streamlit run streamlit_app.py
```

Then upload a text-based PDF, index it, summarize it, and ask questions. Expand
the retrieved context preview to inspect the chunks used by the model.

## Run The FastAPI Service

```powershell
uvicorn src.api:app --reload
```

Useful endpoints:

- `GET /health`
- `POST /documents/upload`
- `POST /summary`
- `POST /query`

## Use ChromaDB Locally

ChromaDB is the default backend:

```env
VECTOR_DB=chroma
CHROMA_PATH=.chroma
```

## Use Pinecone

Set these values in `.env`:

```env
VECTOR_DB=pinecone
PINECONE_API_KEY=your-key
PINECONE_INDEX_NAME=pdf-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_NAMESPACE=default
EMBEDDING_DIMENSION=384
```

The app creates the Pinecone index if it does not exist.

## Enable Reranking

```env
RERANK_ENABLED=true
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Reranking retrieves a wider candidate set and reorders chunks with a
cross-encoder before generation.

## Run Evaluations

First index a document and copy the document ID shown by the app. Then run:

```powershell
python evals/rag_eval.py --document-id <document_id>
```

The evaluator reports:

- Whether expected pages were retrieved.
- Whether the answer included citations.
- Whether expected answer terms appeared.
- Which pages were retrieved.

## Docker

```powershell
docker build -t document-intelligence-rag .
docker run --env-file .env -p 8501:8501 document-intelligence-rag
```

## Resume Bullets

- Built a production-style RAG document intelligence system using Python,
  LangChain, LangGraph, ChromaDB, Pinecone, FastAPI, Streamlit, and Ollama.
- Designed pluggable vector-store adapters with metadata filtering, page-level
  citations, and local/cloud retrieval backends.
- Implemented a graph-based RAG workflow with retrieval, optional reranking,
  grounded generation, citation attachment, and refusal behavior for
  unsupported answers.
- Added evaluation tooling for retrieval hit rate, citation presence, expected
  answer terms, and retrieved page analysis.
