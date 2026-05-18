"""Retrieval-augmented generation helpers."""

from __future__ import annotations

from typing import Any

import requests

from src.config import AppConfig
from src.embeddings_store import EmbeddingsStore


def _format_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Chunk {idx} | Page {chunk['page_number']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def _call_ollama(prompt: str, config: AppConfig) -> str:
    endpoint = f"{config.ollama_base_url.rstrip('/')}/api/generate"
    payload = {
        "model": config.ollama_model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=config.llm_timeout_seconds)
    except requests.RequestException as exc:
        raise RuntimeError(
            "Could not reach Ollama. Ensure Ollama is running locally and the model is pulled."
        ) from exc

    if response.status_code != 200:
        raise RuntimeError(f"Ollama request failed ({response.status_code}): {response.text}")

    output = response.json().get("response", "").strip()
    if not output:
        raise RuntimeError("Model returned an empty response.")
    return output


class RAGService:
    def __init__(self, config: AppConfig, store: EmbeddingsStore) -> None:
        self.config = config
        self.store = store

    def summarize_document(
        self, collection_name: str, document_id: str
    ) -> tuple[str, list[dict[str, Any]]]:
        retrieved = self.store.retrieve(
            collection_name=collection_name,
            query="Summarize the document",
            top_k=max(self.config.top_k, 8),
            document_id=document_id,
        )
        if not retrieved:
            raise ValueError("No relevant context found for summary.")

        context = _format_context(retrieved)
        prompt = (
            "You are a strict document summarizer.\n"
            "Use only the provided context.\n"
            "If context is insufficient, say exactly: 'Insufficient context from document.'\n\n"
            f"Context:\n{context}\n\n"
            "Task: Summarize the document with key points and a concise conclusion."
        )
        answer = _call_ollama(prompt, self.config)
        return answer, retrieved

    def answer_question(
        self, collection_name: str, document_id: str, question: str
    ) -> tuple[str, list[dict[str, Any]]]:
        retrieved = self.store.retrieve(
            collection_name=collection_name,
            query=question,
            top_k=self.config.top_k,
            document_id=document_id,
        )
        if not retrieved:
            raise ValueError("No relevant context found for this question.")

        context = _format_context(retrieved)
        prompt = (
            "You are a strict question-answering assistant.\n"
            "Answer only from the provided context.\n"
            "If the answer is not present, say exactly: 'Insufficient context from document.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        answer = _call_ollama(prompt, self.config)
        return answer, retrieved
