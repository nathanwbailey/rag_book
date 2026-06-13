"""Read PDFs with PyMuPDF and produce paragraph-aware TextNodes.

Three things this loader does beyond a naive page dump:

1. **Full-TOC structure**: the whole table of contents is walked (not just the
   top level), so a chunk is tagged with the real *chapter* it belongs to even
   when chapters are nested under sections/parts. Each chunk also carries
   ``part`` / ``section`` / ``breadcrumb`` for its place in the hierarchy. The
   chapter level is detected per book (some books number at level 1, others
   nest "Chapter N" under a level-1 "Part"/"Section").

2. **Chapter-level grouping**: all pages of a chapter are accumulated before
   chunking, so chunks cross page boundaries and no sentence is severed at a
   page turn.

3. **Block-aware, cleaned chunking**: PyMuPDF's ``get_text("blocks")`` yields
   paragraph-level units; each is normalized (unicode, de-wrapped line breaks)
   and chunks are sealed at block boundaries so they never cut a paragraph in
   half. The tail of each chunk carries forward as overlap.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import fitz
from llama_index.core.schema import TextNode

from . import config
from .references import parse_chapter_number

# Structural fields are kept off the embedding text (noise) and off the LLM
# metadata view; the synthesizer cites from book/chapter/page directly.
_EXCLUDED_EMBED = ["page", "source", "chapter_num", "part", "section", "breadcrumb"]
_EXCLUDED_LLM = ["source", "chapter_num", "part", "section", "breadcrumb"]

# Approximate character-to-token ratio for English prose, used to convert the
# token-based CHUNK_SIZE / CHUNK_OVERLAP config values into character budgets.
_CHARS_PER_TOKEN = 4.5

_BREADCRUMB_SEP = " ▸ "


def _book_title(doc: fitz.Document, path: Path) -> str:
    meta_title = (doc.metadata or {}).get("title") or ""
    meta_title = meta_title.strip()
    return meta_title if meta_title else path.stem


def _clean(text: str) -> str:
    """Normalize one block's text: fix unicode, drop zero-width chars, and
    flatten the PDF's hard line-wraps (a block is one paragraph, so every
    internal newline is a wrap, not a real break)."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("​", "").replace("\xad", "")  # zero-width space, soft hyphen
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _detect_chapter_level(entries: list[tuple[int, int, str]]) -> int:
    """Pick the TOC level that holds the book's chapters.

    Books differ: some number chapters at level 1 ("1. Foo"), others nest
    "Chapter N" under a level-1 "Part"/"Section". The chapter level is the one
    with the most numbered titles; ties favour the shallower level. Falls back
    to level 1 when nothing is numbered.
    """
    counts: dict[int, int] = {}
    for _, level, title in entries:
        if parse_chapter_number(title) is not None:
            counts[level] = counts.get(level, 0) + 1
    if not counts:
        return 1
    return min(counts, key=lambda lvl: (-counts[lvl], lvl))


def _compose(active: dict[int, str], chapter_level: int) -> dict:
    """Turn the active TOC entry at each level into chapter/part/section."""
    levels = sorted(active)
    chapter = active.get(chapter_level) or active[levels[0]]
    part = active.get(1, "")
    if part == chapter:
        part = ""  # chapter is itself the top level — no separate part
    deepest = levels[-1]
    section = active.get(deepest, "") if deepest > chapter_level else ""
    return {
        "part": part,
        "chapter": chapter,
        "chapter_num": parse_chapter_number(chapter) or 0,
        "section": section,
    }


def _build_page_breadcrumbs(toc: list, page_count: int) -> dict[int, dict] | None:
    """Map each 1-based page to its ``{part, chapter, chapter_num, section}``.

    Sweeps pages in order, tracking the active TOC entry at each level (a new
    entry at one level resets the deeper levels). Returns ``None`` when the PDF
    has no usable TOC.
    """
    entries = [
        (page, level, (title or "").strip())
        for level, title, page in toc
        if page >= 1 and (title or "").strip()
    ]
    if not entries:
        return None
    entries.sort(key=lambda e: e[0])  # by page; stable, so same-page keeps TOC order
    chapter_level = _detect_chapter_level(entries)

    page_meta: dict[int, dict] = {}
    active: dict[int, str] = {}
    idx = 0
    for page_no in range(1, page_count + 1):
        while idx < len(entries) and entries[idx][0] == page_no:
            _, level, title = entries[idx]
            active[level] = title
            for deeper in [lvl for lvl in active if lvl > level]:
                del active[deeper]
            idx += 1
        if active:
            page_meta[page_no] = _compose(active, chapter_level)
    return page_meta


def _is_noise_block(text: str, y0: float, y1: float, page_height: float) -> bool:
    """True for empty blocks, bare page numbers, and short blocks in the
    top/bottom margin (running headers/footers). ``text`` is already cleaned."""
    if not text:
        return True
    if re.fullmatch(r"\d+", text):  # bare page number, e.g. "42"
        return True
    # Short blocks in the outermost 6% of the page are likely running heads/feet.
    if len(text) < 80 and (y1 <= page_height * 0.06 or y0 >= page_height * 0.94):
        return True
    return False


def _extract_page_blocks(page: fitz.Page, page_no: int) -> list[tuple[int, str]]:
    """Return ``(page_no, cleaned_text)`` pairs from one page in reading order,
    filtered for noise."""
    page_height = page.rect.height
    # Sort top-to-bottom, left-to-right for single-column reading order.
    raw = sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0]))
    out: list[tuple[int, str]] = []
    for b in raw:
        if b[6] != 0:  # non-text (image) block
            continue
        text = _clean(b[4])
        if _is_noise_block(text, b[1], b[3], page_height):
            continue
        out.append((page_no, text))
    return out


def _make_node(
    pending: list[tuple[int, str, str]], overlap: str, overlap_page_no: int, base: dict
) -> TextNode:
    """Build one chunk from accumulated ``(page, section, text)`` blocks."""
    body = "\n\n".join(t for _, _, t in pending)
    text = (overlap + "\n\n" + body).strip() if overlap else body
    start, end = overlap_page_no, pending[-1][0]
    section = pending[0][1]  # section of the chunk's first new block
    crumbs = [c for c in (base.get("part", ""), base["chapter"], section) if c]
    return TextNode(
        text=text,
        metadata={
            **base,
            "page": str(start) if start == end else f"{start} - {end}",
            "section": section,
            "breadcrumb": _BREADCRUMB_SEP.join(crumbs),
        },
        excluded_embed_metadata_keys=_EXCLUDED_EMBED,
        excluded_llm_metadata_keys=_EXCLUDED_LLM,
    )


def _chunk_blocks(
    blocks: list[tuple[int, str, str]],
    base: dict,
    max_chars: int,
    overlap_chars: int,
) -> list[TextNode]:
    """Aggregate ``(page, section, text)`` blocks into ``TextNode`` chunks.

    Chunks are sealed at block boundaries — never mid-paragraph. Each sealed
    chunk's tail is carried forward as an overlap prefix on the next chunk.
    """
    if not blocks:
        return []

    nodes: list[TextNode] = []
    pending: list[tuple[int, str, str]] = []
    pending_chars = 0
    overlap = ""
    # Page of the overlap's last block — used as the next node's start page.
    overlap_page_no = blocks[0][0]

    for page_no, section, text in blocks:
        if pending_chars + len(text) > max_chars and pending:
            nodes.append(_make_node(pending, overlap, overlap_page_no, base))
            body = "\n\n".join(t for _, _, t in pending)
            overlap = body[-overlap_chars:] if len(body) > overlap_chars else body
            overlap_page_no = pending[-1][0]  # next node starts on the overlap's last page
            pending = []
            pending_chars = 0
        pending.append((page_no, section, text))
        pending_chars += len(text)

    if pending:
        nodes.append(_make_node(pending, overlap, overlap_page_no, base))

    return nodes


def load_pdf(path: Path) -> list[TextNode]:
    """Load one PDF into paragraph-aware ``TextNode``s grouped by chapter."""
    doc = fitz.open(path)
    try:
        title = _book_title(doc, path)
        page_meta = _build_page_breadcrumbs(doc.get_toc(), doc.page_count)
        fallback_chapter = "Front matter" if page_meta is not None else "Full text"

        # Group blocks by chapter, in reading order (dict preserves first-seen
        # order). ``base`` holds the chapter-constant metadata; ``section`` is
        # threaded per block since it can vary within a chapter.
        chapters: dict[str, dict] = {}
        for i, page in enumerate(doc):
            page_no = i + 1
            blocks = _extract_page_blocks(page, page_no)
            if not blocks:
                continue
            meta = (page_meta or {}).get(page_no) or {
                "part": "",
                "chapter": fallback_chapter,
                "chapter_num": 0,
                "section": "",
            }
            chapter = meta["chapter"]
            entry = chapters.setdefault(
                chapter,
                {
                    "base": {
                        "book_title": title,
                        "part": meta["part"],
                        "chapter": chapter,
                        "chapter_num": meta["chapter_num"],
                        "source": path.name,
                    },
                    "blocks": [],
                },
            )
            for page_no_, text in blocks:
                entry["blocks"].append((page_no_, meta["section"], text))

        max_chars = int(config.CHUNK_SIZE * _CHARS_PER_TOKEN)
        overlap_chars = int(config.CHUNK_OVERLAP * _CHARS_PER_TOKEN)

        nodes: list[TextNode] = []
        for entry in chapters.values():
            nodes.extend(_chunk_blocks(entry["blocks"], entry["base"], max_chars, overlap_chars))
        return nodes
    finally:
        doc.close()


def load_books(books_dir: Path) -> list[TextNode]:
    """Load every ``*.pdf`` under ``books_dir`` into paragraph-aware ``TextNode``s."""
    nodes: list[TextNode] = []
    for pdf_path in sorted(books_dir.glob("*.pdf")):
        nodes.extend(load_pdf(pdf_path))
    return nodes
