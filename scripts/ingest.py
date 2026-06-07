"""CLI entry point: ``uv run python -m scripts.ingest``.

Builds the index from PDFs in ``books/``. Makes no Claude API calls, so it runs
fine with ANTHROPIC_API_KEY unset.
"""

from rag_book.ingest import ingest


def main() -> None:
    summary = ingest()
    print(
        f"Indexed {summary['num_books']} book(s), "
        f"{summary['num_pages']} page(s), {summary['num_chunks']} chunk(s)."
    )
    for book in summary["books"]:
        print(f"  - {book}")


if __name__ == "__main__":
    main()
