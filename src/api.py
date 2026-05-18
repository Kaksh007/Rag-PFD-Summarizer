"""FastAPI service for document ingestion and RAG queries."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.chunking import split_pages_into_chunks
from src.config import AppConfig, build_collection_name
from src.embeddings_store import EmbeddingsStore
from src.pdf_ingest import extract_pdf_pages
from src.rag import RAGService


app = FastAPI(
    title="Document Intelligence RAG API",
    description="PDF ingestion, vector indexing, grounded Q&A, and summaries.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    document_id: str
    question: str


class SummaryRequest(BaseModel):
    document_id: str


class RAGResponse(BaseModel):
    answer: str
    retrieved: list[dict[str, Any]]


def _doc_id_from_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


@lru_cache(maxsize=1)
def _services() -> tuple[AppConfig, EmbeddingsStore, RAGService]:
    config = AppConfig()
    config.validate()
    store = EmbeddingsStore(config)
    return config, store, RAGService(config, store)


@app.get("/health")
def health() -> dict[str, str]:
    config, _, _ = _services()
    return {"status": "ok", "vector_db": config.vector_db}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, Any]:
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")

    config, store, _ = _services()
    pdf_bytes = await file.read()
    document_id = _doc_id_from_bytes(pdf_bytes)
    collection_name = build_collection_name(config, document_id)

    try:
        pages = extract_pdf_pages(pdf_bytes)
        chunks = split_pages_into_chunks(
            pages=pages,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        indexed_count = store.index_chunks(collection_name, document_id, chunks)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "document_id": document_id,
        "collection_name": collection_name,
        "pages": len(pages),
        "chunks_indexed": indexed_count,
    }


@app.post("/summary", response_model=RAGResponse)
def summarize(request: SummaryRequest) -> RAGResponse:
    config, _, rag = _services()
    collection_name = build_collection_name(config, request.document_id)
    try:
        answer, retrieved = rag.summarize_document(collection_name, request.document_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RAGResponse(answer=answer, retrieved=retrieved)


@app.post("/query", response_model=RAGResponse)
def query(request: QueryRequest) -> RAGResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question is required.")

    config, _, rag = _services()
    collection_name = build_collection_name(config, request.document_id)
    try:
        answer, retrieved = rag.answer_question(
            collection_name=collection_name,
            document_id=request.document_id,
            question=request.question.strip(),
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RAGResponse(answer=answer, retrieved=retrieved)
