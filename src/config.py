"""Configuration helpers for the PDF RAG app."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    app_mode: str = os.getenv("APP_MODE", "LOCAL_DEV")
    chroma_path: str = os.getenv("CHROMA_PATH", ".chroma")
    collection_prefix: str = os.getenv("COLLECTION_PREFIX", "pdf_rag")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    def validate(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be > 0")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP must be >= 0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if self.top_k <= 0:
            raise ValueError("TOP_K must be > 0")


def build_collection_name(config: AppConfig, session_or_doc_id: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in session_or_doc_id.lower())
    return f"{config.collection_prefix}_{safe[:48]}"
