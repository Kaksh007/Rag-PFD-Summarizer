"""Vector database adapters for local ChromaDB and managed Pinecone."""

from __future__ import annotations

import os
import shutil
import warnings
from typing import Any, Protocol

from src.config import AppConfig


class VectorStore(Protocol):
    """Common interface used by the RAG pipeline."""

    def index_chunks(
        self,
        collection_name: str,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        """Persist chunk text, embeddings, and metadata."""

    def retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        document_id: str,
    ) -> list[dict[str, Any]]:
        """Return the most relevant chunks for a document-scoped query."""


class SentenceTransformerEmbedder:
    """Small embedding wrapper shared by all vector-store backends."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


def _open_persistent_chroma(path: str) -> Any:
    """Open Chroma, resetting incompatible/corrupt local state when needed."""
    resolved = os.path.abspath(path)
    os.makedirs(resolved, exist_ok=True)

    def _client() -> Any:
        import chromadb

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
                f"Chroma could not open {resolved!r}. Removing it and creating a new "
                "store; re-index your PDFs.",
                UserWarning,
                stacklevel=2,
            )
            shutil.rmtree(resolved, ignore_errors=True)
            os.makedirs(resolved, exist_ok=True)
            return _client()
        raise


class ChromaVectorStore:
    """Local persistent ChromaDB implementation."""

    def __init__(self, config: AppConfig, embedder: SentenceTransformerEmbedder) -> None:
        self.config = config
        self.embedder = embedder
        self.client = _open_persistent_chroma(config.chroma_path)

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
        embeddings = self.embedder.embed(texts)
        ids = [f"{document_id}_{c['chunk_id']}" for c in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "page_number": c["page_number"],
                "chunk_index": c["chunk_index"],
                "chunk_id": c["chunk_id"],
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
        query_embedding = self.embedder.embed([query])[0]
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
                    "id": meta.get("chunk_id", f"hit_{idx}"),
                    "text": doc,
                    "page_number": meta.get("page_number", "?"),
                    "chunk_index": meta.get("chunk_index", "?"),
                    "distance": distance,
                    "score": None if distance is None else 1 / (1 + float(distance)),
                    "vector_store": "chroma",
                }
            )
        return hits


class PineconeVectorStore:
    """Managed Pinecone implementation using the current `pinecone` SDK."""

    def __init__(self, config: AppConfig, embedder: SentenceTransformerEmbedder) -> None:
        self.config = config
        self.embedder = embedder
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError as exc:
            raise RuntimeError(
                "Pinecone support requires the `pinecone` package. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc

        self.pc = Pinecone(api_key=config.pinecone_api_key)
        existing = self.pc.list_indexes().names()
        if config.pinecone_index_name not in existing:
            self.pc.create_index(
                name=config.pinecone_index_name,
                dimension=config.embedding_dimension,
                metric=config.pinecone_metric,
                spec=ServerlessSpec(
                    cloud=config.pinecone_cloud,
                    region=config.pinecone_region,
                ),
            )
        self.index = self.pc.Index(config.pinecone_index_name)

    def index_chunks(
        self,
        collection_name: str,
        document_id: str,
        chunks: list[dict[str, Any]],
    ) -> int:
        if not chunks:
            raise ValueError("No chunks to index.")

        namespace = self._namespace(collection_name)
        self.index.delete(
            filter={"document_id": {"$eq": document_id}},
            namespace=namespace,
        )

        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed(texts)
        vectors = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            vectors.append(
                {
                    "id": f"{document_id}_{chunk['chunk_id']}",
                    "values": embedding,
                    "metadata": {
                        "document_id": document_id,
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "chunk_id": chunk["chunk_id"],
                        "text": chunk["text"],
                    },
                }
            )

        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            self.index.upsert(vectors=vectors[start : start + batch_size], namespace=namespace)
        return len(chunks)

    def retrieve(
        self,
        collection_name: str,
        query: str,
        top_k: int,
        document_id: str,
    ) -> list[dict[str, Any]]:
        namespace = self._namespace(collection_name)
        query_embedding = self.embedder.embed([query])[0]
        response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            include_metadata=True,
            filter={"document_id": {"$eq": document_id}},
        )

        hits: list[dict[str, Any]] = []
        for match in response.matches:
            meta = match.metadata or {}
            text = meta.get("text", "")
            if not text:
                continue
            hits.append(
                {
                    "id": meta.get("chunk_id", match.id),
                    "text": text,
                    "page_number": meta.get("page_number", "?"),
                    "chunk_index": meta.get("chunk_index", "?"),
                    "distance": None,
                    "score": float(match.score) if match.score is not None else None,
                    "vector_store": "pinecone",
                }
            )
        return hits

    def _namespace(self, collection_name: str) -> str:
        if self.config.pinecone_namespace:
            return f"{self.config.pinecone_namespace}-{collection_name}"
        return collection_name


def build_vector_store(config: AppConfig) -> VectorStore:
    """Factory for the configured vector database backend."""
    embedder = SentenceTransformerEmbedder(config.embedding_model)
    if config.vector_db == "pinecone":
        return PineconeVectorStore(config, embedder)
    return ChromaVectorStore(config, embedder)
