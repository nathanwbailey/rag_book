"""CLI entry point: ``uv run python -m scripts.ingest``.

Builds the index from PDFs in ``books/``. Makes no LLM calls, so it runs
fine without OPENROUTER_API_KEY.
"""

from rag_book.ingest import ingest


def main() -> None:
    summary = ingest()
    print(
        f"Indexed {summary['num_books']} book(s), "
        f"{summary['num_pages']} page(s), {summary['num_chunks']} chunk(s), "
        f"{summary['num_edges']} reference link(s)."
    )
    for book in summary["books"]:
        print(f"  - {book}")


if __name__ == "__main__":
    main()
