"""LangGraph-compatible RAG workflow with citations and optional reranking."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Literal, TypedDict

from src.config import AppConfig

if TYPE_CHECKING:
    from src.embeddings_store import EmbeddingsStore


TaskType = Literal["summary", "question"]
GenerateFn = Callable[[str, AppConfig], str]


class RAGState(TypedDict, total=False):
    collection_name: str
    document_id: str
    task: TaskType
    question: str
    query: str
    retrieved: list[dict[str, Any]]
    answer: str
    prompt: str


def format_context(chunks: list[dict[str, Any]]) -> str:
    blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Source {idx} | Page {chunk['page_number']} | Chunk "
            f"{chunk['chunk_index']}]\n{chunk['text']}"
        )
    return "\n\n".join(blocks)


def format_sources(chunks: list[dict[str, Any]]) -> str:
    sources = []
    seen = set()
    for idx, chunk in enumerate(chunks, start=1):
        key = (chunk.get("page_number"), chunk.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        score = chunk.get("rerank_score", chunk.get("score"))
        score_text = f", score={score:.3f}" if isinstance(score, (int, float)) else ""
        sources.append(
            f"[{idx}] Page {chunk.get('page_number')}, "
            f"chunk {chunk.get('chunk_index')}{score_text}"
        )
    return "\n".join(sources)


class Reranker:
    """Optional cross-encoder reranker for stronger retrieval precision."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def rerank(self, query: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.config.rerank_enabled or len(chunks) < 2:
            return chunks

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            warnings.warn("RERANK_ENABLED=true but sentence-transformers is unavailable.")
            return chunks

        try:
            if self._model is None:
                self._model = CrossEncoder(self.config.rerank_model)
            pairs = [(query, chunk["text"]) for chunk in chunks]
            scores = self._model.predict(pairs)
            rescored = []
            for chunk, score in zip(chunks, scores, strict=True):
                updated = dict(chunk)
                updated["rerank_score"] = float(score)
                rescored.append(updated)
            return sorted(rescored, key=lambda item: item["rerank_score"], reverse=True)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Reranking failed; using vector order instead: {exc}")
            return chunks


class RAGWorkflow:
    """Small graph orchestration layer around retrieval and generation."""

    def __init__(self, config: AppConfig, store: "EmbeddingsStore", generate_fn: GenerateFn) -> None:
        self.config = config
        self.store = store
        self.generate_fn = generate_fn
        self.reranker = Reranker(config)
        self.graph = self._build_graph()

    def invoke(
        self,
        *,
        collection_name: str,
        document_id: str,
        task: TaskType,
        question: str = "",
    ) -> RAGState:
        query = question if task == "question" else "Summarize the document"
        state: RAGState = {
            "collection_name": collection_name,
            "document_id": document_id,
            "task": task,
            "question": question,
            "query": query,
        }
        if self.graph is not None:
            return self.graph.invoke(state)
        return self._run_sequential(state)

    def _build_graph(self) -> Any | None:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None

        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("rerank", self._rerank)
        workflow.add_node("generate", self._generate)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()

    def _run_sequential(self, state: RAGState) -> RAGState:
        state = self._retrieve(state)
        state = self._rerank(state)
        return self._generate(state)

    def _retrieve(self, state: RAGState) -> RAGState:
        base_k = max(self.config.top_k, 8) if state["task"] == "summary" else self.config.top_k
        expanded_k = max(base_k, 10) if self.config.rerank_enabled else base_k
        retrieved = self.store.retrieve(
            collection_name=state["collection_name"],
            query=state["query"],
            top_k=expanded_k,
            document_id=state["document_id"],
        )
        if not retrieved:
            raise ValueError("No relevant context found for this request.")
        return {**state, "retrieved": retrieved}

    def _rerank(self, state: RAGState) -> RAGState:
        retrieved = self.reranker.rerank(state["query"], state["retrieved"])
        return {**state, "retrieved": retrieved[: self.config.top_k]}

    def _generate(self, state: RAGState) -> RAGState:
        prompt = self._build_prompt(state)
        answer = self.generate_fn(prompt, self.config)
        sources = format_sources(state["retrieved"])
        if sources and "Sources:" not in answer:
            answer = f"{answer}\n\nSources:\n{sources}"
        return {**state, "prompt": prompt, "answer": answer}

    def _build_prompt(self, state: RAGState) -> str:
        context = format_context(state["retrieved"])
        if state["task"] == "summary":
            task_text = (
                "Summarize the document with the main thesis, key points, and a concise "
                "conclusion. Cite source numbers where useful."
            )
        else:
            task_text = f"Question: {state['question']}\nAnswer with source citations."

        template = (
            "You are a strict document intelligence assistant.\n"
            "Use only the provided context.\n"
            "If the answer is not present, say exactly: 'Insufficient context from document.'\n"
            "Do not invent facts outside the retrieved sources.\n\n"
            "Context:\n{context}\n\n"
            "Task: {task}\n"
            "Answer:"
        )
        try:
            from langchain_core.prompts import PromptTemplate

            prompt = PromptTemplate.from_template(template)
            return prompt.format(context=context, task=task_text)
        except ImportError:
            return template.format(context=context, task=task_text)
