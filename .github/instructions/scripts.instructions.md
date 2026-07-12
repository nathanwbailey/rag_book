---
applyTo: "scripts/**/*.py"
---

# scripts Directory Instructions

This directory contains thin command-line entry points that wrap the package code. Keep changes here minimal and focused on invocation behavior.

## Files in this directory

- `__init__.py`: Package marker so `python -m scripts.<name>` works.
  - I/O: none.
- `ingest.py`: CLI entry point for rebuilding the book index.
  - I/O: reads PDFs from `books/`, calls the ingest pipeline, and prints a short summary of indexed books, chunks, and links.