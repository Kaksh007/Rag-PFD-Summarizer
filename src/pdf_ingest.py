"""PDF ingestion utilities."""

from __future__ import annotations

import io
from typing import Any

from pypdf import PdfReader


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Extract text per page with simple metadata."""
    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty.")

    reader = PdfReader(io.BytesIO(pdf_bytes))
    if not reader.pages:
        raise ValueError("PDF has no pages.")

    pages: list[dict[str, Any]] = []
    for idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        pages.append(
            {
                "page_number": idx,
                "text": text,
                "char_count": len(text),
            }
        )

    non_empty = [p for p in pages if p["text"]]
    if not non_empty:
        raise ValueError("No extractable text found. The PDF may be scanned/image-only.")

    return pages
