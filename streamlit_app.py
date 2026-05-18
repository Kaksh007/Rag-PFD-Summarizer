"""Streamlit entrypoint for PDF RAG Summarizer."""

from __future__ import annotations

import hashlib

import streamlit as st

from src.chunking import split_pages_into_chunks
from src.config import AppConfig, build_collection_name
from src.embeddings_store import EmbeddingsStore
from src.pdf_ingest import extract_pdf_pages
from src.rag import RAGService


def _doc_id_from_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()[:16]


def _show_context(chunks: list[dict]) -> None:
    with st.expander("Retrieved context preview", expanded=False):
        for idx, chunk in enumerate(chunks, start=1):
            distance = chunk.get("distance")
            distance_text = f"{distance:.4f}" if isinstance(distance, (int, float)) else "n/a"
            st.markdown(
                f"**{idx}. Page {chunk['page_number']} (chunk {chunk['chunk_index']}) "
                f"| distance={distance_text}**"
            )
            st.write(chunk["text"])
            st.divider()


def main() -> None:
    st.set_page_config(page_title="PDF RAG Summarizer", page_icon="PDF", layout="wide")
    st.title("PDF RAG Summarizer")
    st.caption("Upload a PDF, index it, then generate grounded summary and Q&A answers.")

    config = AppConfig()
    config.validate()
    store = EmbeddingsStore(config)
    rag = RAGService(config, store)

    if "indexed" not in st.session_state:
        st.session_state.indexed = False
        st.session_state.document_id = ""
        st.session_state.collection_name = ""
        st.session_state.last_retrieved = []

    st.subheader("1) Upload PDF")
    uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_pdf:
        pdf_bytes = uploaded_pdf.getvalue()
        document_id = _doc_id_from_bytes(pdf_bytes)
        collection_name = build_collection_name(config, document_id)

        st.info(f"Document ID: `{document_id}`")
        if st.button("Index PDF", type="primary"):
            try:
                pages = extract_pdf_pages(pdf_bytes)
                chunks = split_pages_into_chunks(
                    pages=pages,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap,
                )
                if not chunks:
                    raise ValueError("No chunks generated from PDF text.")

                indexed_count = store.index_chunks(
                    collection_name=collection_name,
                    document_id=document_id,
                    chunks=chunks,
                )
                st.session_state.indexed = True
                st.session_state.document_id = document_id
                st.session_state.collection_name = collection_name
                st.success(f"Indexing complete: {indexed_count} chunks stored.")
            except Exception as exc:  # noqa: BLE001
                st.session_state.indexed = False
                st.error(f"Indexing failed: {exc}")

    st.subheader("2) Generate Summary")
    if st.button("Summarize document", disabled=not st.session_state.indexed):
        try:
            summary, retrieved = rag.summarize_document(
                collection_name=st.session_state.collection_name,
                document_id=st.session_state.document_id,
            )
            st.session_state.last_retrieved = retrieved
            st.markdown("### Summary")
            st.write(summary)
            _show_context(retrieved)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Summary failed: {exc}")

    st.subheader("3) Ask a Question")
    question = st.text_input(
        "Ask about the uploaded PDF",
        placeholder="e.g., What methodology is used in this paper?",
    )
    if st.button("Get answer", disabled=not st.session_state.indexed or not question.strip()):
        try:
            answer, retrieved = rag.answer_question(
                collection_name=st.session_state.collection_name,
                document_id=st.session_state.document_id,
                question=question.strip(),
            )
            st.session_state.last_retrieved = retrieved
            st.markdown("### Answer")
            st.write(answer)
            _show_context(retrieved)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Q&A failed: {exc}")


if __name__ == "__main__":
    main()
