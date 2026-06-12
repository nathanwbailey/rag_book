"""Build a cross-reference graph between chunks — no LLM, no new deps.

A chunk that says "see Chapter 5" or "as on p. 42" is linked to the chunks that
make up that chapter / page (within the same book). The graph is a plain
adjacency map keyed by ``node_id`` so it serializes straight to JSON and can be
followed one hop at query time.

Chapter references are resolved via the book's *printed* chapter numbers, which
are parsed from the numbered chapter titles (e.g. "1. The Manifestations of
TMS"). This avoids matching against our internal ``chapter_num``, which also
counts front matter (copyright/preface/…) and so won't line up with the book's
own numbering. Page references map directly onto the PDF page number and are
therefore best-effort — printed page N rarely equals PDF page N.
"""

from __future__ import annotations

import re
from collections import defaultdict

# "Chapter 5", "chapters 5" — capture the number.
_CHAPTER_RE = re.compile(r"\bchapters?\s+(\d{1,4})\b", re.IGNORECASE)
# "p. 42", "pp. 42", "p42", "page 42", "pages 42" — capture the number.
_PAGE_RE = re.compile(r"\b(?:pp?\.?\s*|pages?\s+)(\d{1,4})\b", re.IGNORECASE)
# Leading printed chapter number in a chapter title: "1. Foo", "2 Foo".
_TITLE_NUM_RE = re.compile(r"^\s*(\d{1,4})[.\s]")


def extract_references(text: str) -> list[tuple[str, int]]:
    """Return de-duplicated ``("chapter"|"page", N)`` references found in text."""
    found: set[tuple[str, int]] = set()
    for match in _CHAPTER_RE.finditer(text):
        found.add(("chapter", int(match.group(1))))
    for match in _PAGE_RE.finditer(text):
        found.add(("page", int(match.group(1))))
    return sorted(found)


def build_chapter_number_map(chapters: list[str]) -> dict[int, str]:
    """Map a book's printed chapter number -> our chapter title.

    Only titles that start with a number contribute, so unnumbered front matter
    (COPYRIGHT, PREFACE, …) is ignored. First title wins on collisions.
    """
    mapping: dict[int, str] = {}
    for title in chapters:
        match = _TITLE_NUM_RE.match(title)
        if match:
            mapping.setdefault(int(match.group(1)), title)
    return mapping


def build_reference_graph(nodes) -> dict[str, list[dict]]:
    """Build ``node_id -> [edge, ...]`` from in-text chapter/page references.

    Each edge is ``{"node_id", "ref_type", "ref_value"}``. Resolution is scoped
    to the source node's own book; self-edges (same page or same chapter) are
    dropped. Makes no LLM calls.
    """
    # Per-book lookups: page -> node_ids, chapter title -> node_ids, and the
    # ordered set of chapter titles (for the printed-number map).
    page_index: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    chapter_index: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    book_chapters: dict[str, dict[int, str]] = defaultdict(dict)

    for node in nodes:
        meta = node.metadata or {}
        book = meta.get("book_title")
        if not book:
            continue
        page = meta.get("page")
        chapter = meta.get("chapter")
        chapter_num = meta.get("chapter_num")
        if isinstance(page, int):
            page_index[book][page].append(node.node_id)
        if chapter:
            chapter_index[book][chapter].append(node.node_id)
            if isinstance(chapter_num, int):
                book_chapters[book][chapter_num] = chapter

    # printed chapter number -> our chapter title, per book.
    printed_map: dict[str, dict[int, str]] = {
        book: build_chapter_number_map([title for _, title in sorted(chapters.items())])
        for book, chapters in book_chapters.items()
    }

    graph: dict[str, list[dict]] = {}
    for node in nodes:
        meta = node.metadata or {}
        book = meta.get("book_title")
        if not book:
            continue
        src_page = meta.get("page")
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
                if value == src_page:
                    continue  # self-page
                targets = page_index[book].get(value, [])
            for target_id in targets:
                if target_id in seen:
                    continue
                seen.add(target_id)
                edges.append({"node_id": target_id, "ref_type": ref_type, "ref_value": value})

        if edges:
            graph[node.node_id] = edges
    return graph
