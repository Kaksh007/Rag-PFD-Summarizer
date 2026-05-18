"""Retrieval-augmented generation service."""

from __future__ import annotations

import requests

from src.config import AppConfig
from src.embeddings_store import EmbeddingsStore
from src.rag_workflow import RAGWorkflow


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
        self.workflow = RAGWorkflow(config, store, _call_ollama)

    def summarize_document(
        self, collection_name: str, document_id: str
    ) -> tuple[str, list[dict[str, Any]]]:
        result = self.workflow.invoke(
            collection_name=collection_name,
            document_id=document_id,
            task="summary",
        )
        return result["answer"], result["retrieved"]

    def answer_question(
        self, collection_name: str, document_id: str, question: str
    ) -> tuple[str, list[dict[str, Any]]]:
        result = self.workflow.invoke(
            collection_name=collection_name,
            document_id=document_id,
            task="question",
            question=question,
        )
        return result["answer"], result["retrieved"]
