"""Lightweight evaluation helpers for RAG demos and regression checks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    question: str
    retrieval_hit: bool
    citation_present: bool
    answer_contains_expected_terms: bool
    retrieved_pages: list[int]


def _normalize_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9]{3,}", text.lower()))


def evaluate_answer(
    *,
    question: str,
    answer: str,
    retrieved: list[dict[str, Any]],
    expected_answer_terms: list[str] | None = None,
    expected_pages: list[int] | None = None,
) -> EvaluationResult:
    retrieved_pages = [
        int(chunk["page_number"])
        for chunk in retrieved
        if isinstance(chunk.get("page_number"), int)
    ]
    expected_pages = expected_pages or []
    expected_answer_terms = expected_answer_terms or []

    answer_terms = _normalize_terms(answer)
    required_terms = {term.lower() for term in expected_answer_terms}
    return EvaluationResult(
        question=question,
        retrieval_hit=not expected_pages or bool(set(expected_pages) & set(retrieved_pages)),
        citation_present="Sources:" in answer and "Page" in answer,
        answer_contains_expected_terms=not required_terms or required_terms.issubset(answer_terms),
        retrieved_pages=retrieved_pages,
    )
