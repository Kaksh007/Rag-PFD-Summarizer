"""Configuration helpers for the PDF RAG app."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    app_mode: str = os.getenv("APP_MODE", "LOCAL_DEV")
    vector_db: str = os.getenv("VECTOR_DB", "chroma").lower()
    chroma_path: str = os.getenv("CHROMA_PATH", ".chroma")
    collection_prefix: str = os.getenv("COLLECTION_PREFIX", "pdf_rag")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    rerank_enabled: bool = os.getenv("RERANK_ENABLED", "false").lower() == "true"
    rerank_model: str = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    citation_style: str = os.getenv("CITATION_STYLE", "page_chunk")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "pdf-rag")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")
    pinecone_namespace: str = os.getenv("PINECONE_NAMESPACE", "default")
    pinecone_metric: str = os.getenv("PINECONE_METRIC", "cosine")

    def validate(self) -> None:
        if self.vector_db not in {"chroma", "pinecone"}:
            raise ValueError("VECTOR_DB must be either 'chroma' or 'pinecone'")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be > 0")
        if self.embedding_dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be > 0")
        if self.vector_db == "pinecone" and not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required when VECTOR_DB=pinecone")


def build_collection_name(config: AppConfig, session_or_doc_id: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in session_or_doc_id.lower())
    return f"{config.collection_prefix}_{safe[:48]}"
