"""Backward-compatible vector-store entrypoint.

The original app imported `EmbeddingsStore` directly. It now delegates to the
pluggable vector-store factory so existing UI code keeps working while the
project supports both ChromaDB and Pinecone.
"""

from __future__ import annotations

from typing import Any

from src.config import AppConfig
from src.vector_stores import VectorStore, build_vector_store


class EmbeddingsStore:
    def __init__(self, config: AppConfig) -> None:
        self.backend: VectorStore = build_vector_store(config)

    def index_chunks(
        self,
        collection_name: str,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        return self.backend.index_chunks(collection_name, document_id, chunks)

    def retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        document_id: str,
    ) -> list[dict[str, Any]]:
        return self.backend.retrieve(collection_name, query, top_k, document_id)
