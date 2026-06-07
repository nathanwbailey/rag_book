"""Ingestion pipeline: PDFs -> chapter-aware chunks -> Chroma + manifest.

This path makes NO LLM/Anthropic calls. Embeddings run on the local BGE model.
"""

from __future__ import annotations

import json

from llama_index.core.node_parser import SentenceSplitter

from . import config
from .index import build_index, reset_collection
from .pdf_loader import load_books


def _write_manifest(documents) -> dict[str, list[str]]:
    """Write ``book -> ordered chapter list`` so tools avoid scanning the store."""
    books: dict[str, set[tuple[int, str]]] = {}
    for doc in documents:
        meta = doc.metadata
        books.setdefault(meta["book_title"], set()).add((meta["chapter_num"], meta["chapter"]))

    manifest = {
        book: [chapter for _, chapter in sorted(chapters)]
        for book, chapters in sorted(books.items())
    }
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def ingest() -> dict:
    """(Re)build the index from every PDF in ``books/``. Returns a summary."""
    config.configure_embeddings()

    documents = load_books(config.BOOKS_DIR)
    if not documents:
        raise SystemExit(f"No PDFs found in {config.BOOKS_DIR}. Drop some .pdf files there first.")

    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)

    reset_collection()
    build_index(nodes)
    manifest = _write_manifest(documents)

    return {
        "books": list(manifest.keys()),
        "num_books": len(manifest),
        "num_pages": len(documents),
        "num_chunks": len(nodes),
    }
