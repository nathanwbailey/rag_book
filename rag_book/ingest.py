"""Ingestion pipeline: PDFs -> paragraph-aware chunks -> Chroma + manifest.

This path makes NO LLM calls. Embeddings run on the local BGE model.
"""

from __future__ import annotations

import json

from . import config
from .index import build_index, get_embeddings, reset_collection, save_reference_graph
from .pdf_loader import load_books
from .references import build_reference_graph, build_semantic_bridges


def _write_manifest(nodes) -> dict[str, list[str]]:
    """Write ``book -> ordered chapter list`` so tools avoid scanning the store.

    Nodes arrive in reading order, so chapters are listed in first-seen order.
    """
    books: dict[str, list[str]] = {}
    for node in nodes:
        meta = node.metadata
        chapters = books.setdefault(meta["book_title"], [])
        if meta["chapter"] not in chapters:
            chapters.append(meta["chapter"])

    manifest = dict(sorted(books.items()))
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))
    return manifest


def ingest() -> dict:
    """(Re)build the index from every PDF in ``books/``. Returns a summary."""
    config.configure_embeddings()

    nodes = load_books(config.BOOKS_DIR)
    if not nodes:
        raise SystemExit(f"No PDFs found in {config.BOOKS_DIR}. Drop some .pdf files there first.")

    reset_collection()
    build_index(nodes)
    manifest = _write_manifest(nodes)

    # Build the cross-reference graph from the same nodes (no LLM calls).
    graph = build_reference_graph(nodes)
    num_xref = sum(len(edges) for edges in graph.values())

    # Cross-book semantic bridges: reuse the index-time embeddings to link each
    # chunk to its closest chunks in other books. Merged into the same graph
    # (cross-reference edges first) so the one-hop expander follows both.
    bridges = build_semantic_bridges(
        nodes,
        get_embeddings(),
        threshold=config.SEMANTIC_BRIDGE_THRESHOLD,
        top_k=config.SEMANTIC_BRIDGE_TOP_K,
    )
    for node_id, edges in bridges.items():
        graph.setdefault(node_id, []).extend(edges)
    num_bridges = sum(len(edges) for edges in bridges.values())

    save_reference_graph(graph)

    return {
        "books": list(manifest.keys()),
        "num_books": len(manifest),
        "num_chunks": len(nodes),
        "num_edges": num_xref,
        "num_bridges": num_bridges,
    }
