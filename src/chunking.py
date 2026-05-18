"""Chunking helpers for retrievable text units."""

from __future__ import annotations

from typing import Any


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


def split_pages_into_chunks(
    pages: list[dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict[str, Any]]:
    """Split each page into overlapping chunks with metadata."""
    chunks: list[dict[str, Any]] = []

    for page in pages:
        page_number = page["page_number"]
        text = page["text"]
        page_chunks = _split_text(text=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk_text in enumerate(page_chunks):
            chunks.append(
                {
                    "chunk_id": f"page_{page_number}_chunk_{idx}",
                    "chunk_index": idx,
                    "page_number": page_number,
                    "text": chunk_text,
                }
            )

    return chunks
