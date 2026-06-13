"""Build a one-hop graph between chunks — no LLM.

Two edge kinds share one adjacency map (keyed by ``node_id``, JSON-serializable,
followed one hop at query time):

- **Cross-references** (``build_reference_graph``): a chunk that says "see
  Chapter 5" or "as on p. 42" is linked to the chunks that make up that chapter
  / page *within the same book* (regex, no embeddings).
- **Semantic bridges** (``build_semantic_bridges``): a chunk is linked to its
  most similar chunks in *other* books, reusing the index-time embeddings.

Chapter references are resolved via the book's *printed* chapter numbers, which
are parsed from the numbered chapter titles (e.g. "1. Introduction" or
"Chapter 1: ..."). This avoids matching against our internal ``chapter_num``, which also
counts front matter (copyright/preface/…) and so won't line up with the book's
own numbering. Page references map directly onto the PDF page number and are
therefore best-effort — printed page N rarely equals PDF page N.
"""

from __future__ import annotations

import re
from collections import defaultdict

import numpy as np

# Structural front matter / back matter (matched case-insensitively on the
# exact chapter title, plus a few prefixes). Excluded from semantic bridges so
# cross-book links connect real content, not two copyright or index pages.
_NOISE_TITLES = frozenset(
    {
        "copyright",
        "contents",
        "index",
        "notes",
        "dedication",
        "epigraph",
        "title page",
        "about the author",
        "about the authors",
        "acknowledgments",
        "acknowledgements",
        "useful resources",
        "discover more",
        "front matter",
        "glossary",
        "bibliography",
    }
)
_NOISE_PREFIXES = (
    "praise for",
    "advance praise",
    "also by",
    "by the same author",
    "further reading",
)

# "Chapter 5", "chapters 5" — capture the number.
_CHAPTER_RE = re.compile(r"\bchapters?\s+(\d{1,4})\b", re.IGNORECASE)
# "p. 42", "pp. 42", "p42", "page 42", "pages 42" — capture the number.
_PAGE_RE = re.compile(r"\b(?:pp?\.?\s*|pages?\s+)(\d{1,4})\b", re.IGNORECASE)
# Printed chapter number at the start of a chapter TITLE — either the bare
# "3. Foo" / "3 Foo" form or the "Chapter 3: Foo" form (books differ).
_TITLE_NUM_RE = re.compile(r"^\s*(?:chapter\s+(\d{1,4})\b|(\d{1,4})[.\s])", re.IGNORECASE)


def parse_chapter_number(title: str) -> int | None:
    """Printed chapter number from a title — ``"3. Foo"`` or ``"Chapter 3: Foo"``
    -> ``3``. Returns ``None`` for unnumbered titles (parts, sections, front
    matter), so callers can tell a real chapter from a section heading."""
    match = _TITLE_NUM_RE.match(title or "")
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def _is_structural_noise(chapter: str) -> bool:
    """True for copyright / index / praise-page style chapters (no real prose)."""
    title = chapter.strip().lower()
    return title in _NOISE_TITLES or any(title.startswith(p) for p in _NOISE_PREFIXES)


def _page_span(page: object) -> tuple[int, int] | None:
    """Parse the ``page`` metadata into an inclusive ``(start, end)`` page range.

    The loader writes ``"18"`` for a single-page chunk or ``"18 - 19"`` when a
    chunk straddles a page boundary; a bare int is also accepted for robustness.
    Returns ``None`` when no page number is present.
    """
    if isinstance(page, int):
        return page, page
    if isinstance(page, str):
        nums = [int(n) for n in re.findall(r"\d+", page)]
        if nums:
            return min(nums), max(nums)
    return None


def extract_references(text: str) -> list[tuple[str, int]]:
    """Return de-duplicated ``("chapter"|"page", N)`` references found in text."""
    found: set[tuple[str, int]] = set()
    for match in _CHAPTER_RE.finditer(text):
        found.add(("chapter", int(match.group(1))))
    for match in _PAGE_RE.finditer(text):
        found.add(("page", int(match.group(1))))
    return sorted(found)


def build_reference_graph(nodes) -> dict[str, list[dict]]:
    """Build ``node_id -> [edge, ...]`` from in-text chapter/page references.

    Each edge is ``{"node_id", "ref_type", "ref_value"}``. Resolution is scoped
    to the source node's own book; self-edges (same page or same chapter) are
    dropped. Makes no LLM calls.
    """
    # Per-book lookups: page -> node_ids, chapter title -> node_ids, and the
    # printed chapter number -> chapter title (the loader already parsed the
    # printed number into ``chapter_num``; 0 means "not a numbered chapter").
    page_index: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    chapter_index: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    printed_map: dict[str, dict[int, str]] = defaultdict(dict)

    for node in nodes:
        meta = node.metadata or {}
        book = meta.get("book_title")
        if not book:
            continue
        chapter = meta.get("chapter")
        chapter_num = meta.get("chapter_num")
        # A chunk can straddle pages; index it under every page in its span so a
        # "p. N" reference resolves whenever N falls inside the chunk's range.
        span = _page_span(meta.get("page"))
        if span:
            for p in range(span[0], span[1] + 1):
                page_index[book][p].append(node.node_id)
        if chapter:
            chapter_index[book][chapter].append(node.node_id)
            if isinstance(chapter_num, int) and chapter_num > 0:
                printed_map[book].setdefault(chapter_num, chapter)

    graph: dict[str, list[dict]] = {}
    for node in nodes:
        meta = node.metadata or {}
        book = meta.get("book_title")
        if not book:
            continue
        src_span = _page_span(meta.get("page"))
        src_chapter = meta.get("chapter")

        edges: list[dict] = []
        seen: set[str] = {node.node_id}
        for ref_type, value in extract_references(node.text or ""):
            if ref_type == "chapter":
                title = printed_map.get(book, {}).get(value)
                if not title or title == src_chapter:
                    continue  # unknown or self-chapter
                targets = chapter_index[book].get(title, [])
            else:  # page
                if src_span and src_span[0] <= value <= src_span[1]:
                    continue  # self-page (reference falls within this chunk's own span)
                targets = page_index[book].get(value, [])
            for target_id in targets:
                if target_id in seen:
                    continue
                seen.add(target_id)
                edges.append({"node_id": target_id, "ref_type": ref_type, "ref_value": value})

        if edges:
            graph[node.node_id] = edges
    return graph


def build_semantic_bridges(
    nodes,
    embeddings: dict[str, list[float]],
    *,
    threshold: float,
    top_k: int,
) -> dict[str, list[dict]]:
    """Link each content chunk to its most similar chunks in *other* books.

    Reuses the index-time embeddings (cosine similarity on L2-normalized
    vectors). For each chunk, keeps up to ``top_k`` cross-book neighbours
    scoring at or above ``threshold``, strongest first. Structural front matter
    is excluded on both ends. Makes no LLM calls.

    Returns ``node_id -> [edge, ...]`` where each edge is
    ``{"node_id", "ref_type": "related", "ref_value": <rounded cosine>}`` —
    the same shape as ``build_reference_graph`` so the two merge into one graph
    and the one-hop expander follows both.
    """
    items = [
        (node.node_id, node.metadata.get("book_title"), embeddings[node.node_id])
        for node in nodes
        if node.node_id in embeddings and not _is_structural_noise(node.metadata.get("chapter", ""))
    ]
    if len(items) < 2:
        return {}

    ids = [it[0] for it in items]
    books = np.array([it[1] for it in items])
    vecs = np.asarray([it[2] for it in items], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

    sims = vecs @ vecs.T
    sims[books[:, None] == books[None, :]] = -1.0  # drop self + same-book pairs

    graph: dict[str, list[dict]] = {}
    for i, node_id in enumerate(ids):
        row = sims[i]
        best = np.argsort(row)[::-1][:top_k]
        edges = [
            {"node_id": ids[j], "ref_type": "related", "ref_value": round(float(row[j]), 3)}
            for j in best
            if row[j] >= threshold
        ]
        if edges:
            graph[node_id] = edges
    return graph
