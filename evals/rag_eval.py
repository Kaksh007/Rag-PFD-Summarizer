"""Run a small RAG evaluation suite against an already indexed document.

Usage:
    python evals/rag_eval.py --document-id <doc_id> --questions evals/sample_questions.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import AppConfig, build_collection_name  # noqa: E402
from src.embeddings_store import EmbeddingsStore  # noqa: E402
from src.evaluation import evaluate_answer  # noqa: E402
from src.rag import RAGService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-id", required=True)
    parser.add_argument("--questions", default="evals/sample_questions.json")
    args = parser.parse_args()

    config = AppConfig()
    config.validate()
    store = EmbeddingsStore(config)
    rag = RAGService(config, store)
    collection_name = build_collection_name(config, args.document_id)

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    results = []
    for item in questions:
        answer, retrieved = rag.answer_question(
            collection_name=collection_name,
            document_id=args.document_id,
            question=item["question"],
        )
        result = evaluate_answer(
            question=item["question"],
            answer=answer,
            retrieved=retrieved,
            expected_answer_terms=item.get("expected_answer_terms", []),
            expected_pages=item.get("expected_pages", []),
        )
        results.append(result.__dict__)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
