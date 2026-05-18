"""Embeddings and ChromaDB persistence layer."""

from __future__ import annotations

import os
import shutil
import warnings
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import AppConfig


def _open_persistent_chroma(path: str) -> Any:
    """Open a persistent Chroma client, resetting the folder if the DB is unreadable.

    Newer Chroma uses a Rust-backed SQLite layout. An older ``.chroma`` directory can
    trigger a Rust panic and misleading errors like "Could not connect to tenant".
    """
    resolved = os.path.abspath(path)
    os.makedirs(resolved, exist_ok=True)

    def _client() -> Any:
        return chromadb.PersistentClient(path=resolved)

    try:
        return _client()
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        text = f"{type(exc).__name__} {exc}".lower()
        recoverable = (
            "panic" in text
            or "pyo3" in text
            or "bindings" in text
            or "default_tenant" in text
            or "could not connect to tenant" in text
        )
        if recoverable and os.path.isdir(resolved):
            warnings.warn(
                f"Chroma could not open {resolved!r} (incompatible or corrupt on-disk "
                "data). Removing it and creating a new store; re-index your PDFs.",
                UserWarning,
                stacklevel=2,
            )
            shutil.rmtree(resolved, ignore_errors=True)
            os.makedirs(resolved, exist_ok=True)
            return _client()
        raise


class EmbeddingsStore:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = _open_persistent_chroma(config.chroma_path)
        self.model = SentenceTransformer(config.embedding_model)

    def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    def index_chunks(
        self,
        collection_name: str,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        if not chunks:
            raise ValueError("No chunks to index.")

        collection = self.client.get_or_create_collection(name=collection_name)
        collection.delete(where={"document_id": document_id})

        texts = [c["text"] for c in chunks]
        embeddings = self._embed(texts)
        ids = [f"{document_id}_{c['chunk_id']}" for c in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
            }
            for c in chunks
        ]

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        document_id: str,
    ) -> list[dict[str, Any]]:
        collection = self.client.get_or_create_collection(name=collection_name)
        query_embedding = self._embed([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where={"document_id": document_id},
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        hits: list[dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            if not doc:
                continue
            meta = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else None
            hits.append(
                {
                    "text": doc,
                    "page_number": meta.get("page_number", "?"),
                    "chunk_index": meta.get("chunk_index", "?"),
                    "distance": distance,
                }
            )
        return hits
