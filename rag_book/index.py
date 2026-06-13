"""Chroma-backed vector index with local BGE embeddings.

Building the index embeds nodes locally and upserts them to a persistent Chroma
collection — no LLM calls. ``get_chapter_chunks`` reads raw chunks back by
metadata (no embedding query) so a whole chapter can be reconstructed in reading
order for summarization.
"""

from __future__ import annotations

import json

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

from . import config


def _client() -> chromadb.ClientAPI:
    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def _collection(client: chromadb.ClientAPI | None = None):
    client = client or _client()
    return client.get_or_create_collection(config.COLLECTION_NAME)


def _vector_store() -> ChromaVectorStore:
    return ChromaVectorStore(chroma_collection=_collection())


def reset_collection() -> None:
    """Drop the collection so a re-ingest is a clean full rebuild (no dupes)."""
    client = _client()
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass  # collection did not exist yet


def build_index(nodes) -> VectorStoreIndex:
    """Embed nodes locally and persist them to Chroma."""
    embed = config.configure_embeddings()
    storage = StorageContext.from_defaults(vector_store=_vector_store())
    return VectorStoreIndex(nodes, storage_context=storage, embed_model=embed)


def load_index() -> VectorStoreIndex:
    """Load the persisted index for querying."""
    embed = config.configure_embeddings()
    return VectorStoreIndex.from_vector_store(_vector_store(), embed_model=embed)


def get_all_nodes() -> list[TextNode]:
    """Return every chunk in the store as ``TextNode``s (text + metadata).

    Chroma holds the nodes (not LlamaIndex's docstore), so BM25 must be built
    from these rather than from ``index.docstore`` (which is empty).
    """
    res = _collection().get(include=["documents", "metadatas"])
    ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    return [
        TextNode(id_=id_, text=text or "", metadata=meta or {})
        for id_, text, meta in zip(ids, docs, metas, strict=True)
    ]


def get_nodes_by_ids(ids: list[str]) -> list[TextNode]:
    """Fetch specific chunks by id (used to follow reference-graph edges)."""
    if not ids:
        return []
    res = _collection().get(ids=ids, include=["documents", "metadatas"])
    got_ids = res.get("ids") or []
    docs = res.get("documents") or []
    metas = res.get("metadatas") or []
    return [
        TextNode(id_=id_, text=text or "", metadata=meta or {})
        for id_, text, meta in zip(got_ids, docs, metas, strict=True)
    ]


def get_embeddings() -> dict[str, list[float]]:
    """Return ``node_id -> embedding`` for every chunk in the store.

    Reuses the vectors already computed at index time (no re-embedding) so the
    semantic-bridge builder can measure chunk-to-chunk similarity.
    """
    res = _collection().get(include=["embeddings"])
    ids = res.get("ids") or []
    embs = res.get("embeddings")
    if embs is None or len(embs) == 0:
        return {}
    return {id_: list(emb) for id_, emb in zip(ids, embs, strict=True)}


def save_reference_graph(graph: dict) -> None:
    """Persist the cross-reference adjacency map as JSON."""
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    config.REFERENCE_GRAPH_PATH.write_text(json.dumps(graph, indent=2))


def load_reference_graph() -> dict:
    """Load the cross-reference graph, or ``{}`` if it has not been built yet."""
    if config.REFERENCE_GRAPH_PATH.exists():
        return json.loads(config.REFERENCE_GRAPH_PATH.read_text())
    return {}


def get_chapter_chunks(book: str, chapter: str) -> list[tuple[dict, str]]:
    """Return ``(metadata, text)`` for every chunk of one chapter, page-ordered.

    Uses Chroma's metadata ``where`` filter directly — no similarity query — so
    the full chapter is returned deterministically.
    """
    res = _collection().get(
        where={"$and": [{"book_title": {"$eq": book}}, {"chapter": {"$eq": chapter}}]}
    )
    metas = res.get("metadatas") or []
    docs = res.get("documents") or []
    pairs = list(zip(metas, docs, strict=True))
    pairs.sort(key=lambda pair: pair[0].get("page", 0))
    return pairs
