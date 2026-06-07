"""Read PDFs with PyMuPDF and map pages to chapters via the embedded TOC.

Produces one LlamaIndex ``Document`` per page, tagged with book/chapter/page
metadata. No LLM calls.
"""

from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from llama_index.core import Document

# Metadata kept out of the embedding text (noise) but some still shown to the LLM
# so it can cite book / chapter / page.
_EXCLUDED_EMBED = ["page", "source", "chapter_num"]
_EXCLUDED_LLM = ["source", "chapter_num"]


def _book_title(doc: fitz.Document, path: Path) -> str:
    meta_title = (doc.metadata or {}).get("title") or ""
    meta_title = meta_title.strip()
    return meta_title if meta_title else path.stem


def _build_page_map(toc: list, page_count: int) -> dict[int, tuple[int, str]] | None:
    """Map each 1-based page number to ``(chapter_num, chapter_title)``.

    Uses only top-level (level 1) TOC entries as chapters. Returns ``None`` when
    the PDF has no usable TOC.
    """
    tops = [(title.strip(), page) for level, title, page in toc if level == 1 and page >= 1]
    if not tops:
        return None
    tops.sort(key=lambda x: x[1])

    page_map: dict[int, tuple[int, str]] = {}
    for idx, (title, start) in enumerate(tops):
        end = tops[idx + 1][1] - 1 if idx + 1 < len(tops) else page_count
        for p in range(start, end + 1):
            page_map[p] = (idx + 1, title)
    return page_map


def load_pdf(path: Path) -> list[Document]:
    """Load one PDF into per-page Documents with chapter metadata."""
    doc = fitz.open(path)
    try:
        title = _book_title(doc, path)
        page_map = _build_page_map(doc.get_toc(), doc.page_count)

        documents: list[Document] = []
        for i, page in enumerate(doc):
            page_no = i + 1
            text = page.get_text("text").strip()
            if not text:
                continue

            if page_map is not None:
                chapter_num, chapter = page_map.get(page_no, (0, "Front matter"))
            else:
                chapter_num, chapter = (1, "Full text")

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "book_title": title,
                        "chapter": chapter,
                        "chapter_num": chapter_num,
                        "page": page_no,
                        "source": path.name,
                    },
                    excluded_embed_metadata_keys=_EXCLUDED_EMBED,
                    excluded_llm_metadata_keys=_EXCLUDED_LLM,
                )
            )
        return documents
    finally:
        doc.close()


def load_books(books_dir: Path) -> list[Document]:
    """Load every ``*.pdf`` under ``books_dir`` into Documents."""
    documents: list[Document] = []
    for pdf_path in sorted(books_dir.glob("*.pdf")):
        documents.extend(load_pdf(pdf_path))
    return documents
