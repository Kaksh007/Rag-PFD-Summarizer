# PDF RAG Streamlit App

A local-first PDF summarizer and Q&A app built with Streamlit, ChromaDB, and Ollama.

## Features

- Upload a PDF and extract text by page.
- Split text into overlapping chunks for retrieval.
- Generate embeddings with `sentence-transformers`.
- Index and query chunks in local ChromaDB.
- Produce grounded summary and Q&A responses using local Ollama model.
- Show retrieved context snippets for transparency.

## Project Structure

- `streamlit_app.py`: Streamlit UI entrypoint.
- `src/config.py`: environment-based app config.
- `src/pdf_ingest.py`: PDF text extraction.
- `src/chunking.py`: chunking logic.
- `src/embeddings_store.py`: embeddings + Chroma persistence/retrieval.
- `src/rag.py`: retrieval + grounded generation prompts.

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Create `.env` from template:
   - Copy `.env.example` to `.env` and adjust values if needed.
4. Install and run Ollama:
   - [https://ollama.com](https://ollama.com)
   - `ollama pull llama3.2:1b` (use `llama3.2:3b` in `.env` only if you have enough free RAM for Ollama)
5. Start app:
   - `streamlit run streamlit_app.py`

## Usage

1. Open the Streamlit app URL.
2. Upload a text-based PDF.
3. Click **Index PDF**.
4. Click **Summarize document** for grounded summary.
5. Enter a question and click **Get answer** for grounded Q&A.
6. Expand **Retrieved context preview** to inspect retrieved chunks.

## Validation Checklist

- Upload/index sample PDF succeeds.
- Summary response is grounded in retrieved chunks.
- Q&A response is grounded and refuses unsupported claims with:
  - `Insufficient context from document.`
- App handles setup errors (missing Ollama/service unavailable) with clear messages.
