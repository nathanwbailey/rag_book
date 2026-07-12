# Copilot Cloud Agent Onboarding

## What this repository does

This is a small Python app for question answering over a personal library of PDF books. It ingests PDFs, chunks them, builds a persistent Chroma-backed index with local embeddings, and serves a Streamlit chat UI that can answer chapter-summary and content questions with citations. The query path can use an OpenRouter-hosted LLM, but ingestion and indexing are fully local.

## High-level repo facts

- Language: Python only.
- Runtime: validated with Python 3.13.7.
- Tooling: uv 0.10.7, ruff.
- Main frameworks and libraries: Streamlit, LlamaIndex, ChromaDB, PyMuPDF, sentence-transformers.
- Repo size/shape: small, single-package application with a few entry points and no test suite or GitHub workflow files in the repo.

## Layout

- [app.py](app.py): Streamlit UI entry point.
- [rag_book/agent.py](rag_book/agent.py): tool-calling agent wrapper.
- [rag_book/engine.py](rag_book/engine.py): shared retrieval/synthesis engine.
- [rag_book/ingest.py](rag_book/ingest.py): PDF ingest pipeline and index rebuild.
- [rag_book/index.py](rag_book/index.py): Chroma persistence helpers.
- [rag_book/pdf_loader.py](rag_book/pdf_loader.py): PDF parsing, TOC/chapter detection, chunking.
- [rag_book/references.py](rag_book/references.py): cross-reference and semantic-bridge graph builders.
- [rag_book/session.py](rag_book/session.py): persisted chat/session state.
- [rag_book/config.py](rag_book/config.py): paths, model names, chunking and retrieval knobs.
- [scripts/ingest.py](scripts/ingest.py): CLI wrapper for indexing.
- [books/](books/): input PDFs.
- [storage/](storage/): generated manifest, reference graph, session state, and Chroma data.
- [pyproject.toml](pyproject.toml): packaging, dependencies, and ruff config.
- [README.md](README.md): authoritative user-facing setup and usage notes.

## Build, run, and validate

Always use uv-wrapped commands. Running python or streamlit directly can hit the wrong interpreter and fail on missing dependencies.

1. Bootstrap: `uv sync`
   - Validated with uv 0.10.7 and Python 3.13.7.
   - Creates or refreshes the local `.venv`.
   - If it appears to stall, it is usually downloading packages for the first time.

2. Lint: `uv run ruff check .`
   - Validated and passed.
   - This is the fastest reliable repo-wide check.

3. Ingest/index: `uv run python -m scripts.ingest`
   - Validated and completed successfully on the current repo state.
   - This command does not need `OPENROUTER_API_KEY`.
   - First run may print a Hugging Face rate-limit warning while local models download; that is expected.
   - In this repo state it indexed 4 books and rebuilt the Chroma store under `storage/chroma/`.

4. Run the app: `uv run streamlit run app.py`
   - This is the normal runtime entry point from the README.
   - Querying requires `OPENROUTER_API_KEY` in `.env` or the shell.
   - If the index is missing, the app auto-runs ingestion before starting.

5. Formatting: `uv run ruff format .`
   - Use before landing code changes.
   - No separate formatter config exists beyond ruff in `pyproject.toml`.

## Validation expectations

- There are no repo workflows or test files to mirror, so local validation is the main gate.
- Prefer `uv run ruff check .` for quick confidence after edits.
- If you touch ingestion, PDF loading, or indexing, also run `uv run python -m scripts.ingest`.
- If you touch the chat/UI path, run the Streamlit app after indexing and confirm it starts without stack traces.

## Important repository behaviors

- Ingestion is local-only and builds the manifest, reference graph, and Chroma index from PDFs in `books/`.
- Querying is the only path that needs OpenRouter credentials.
- `storage/` and `books/*.pdf` are generated/local data; do not treat them as source files unless the task explicitly concerns data.
- `uv run` is the safest default for every command in this repo.
- The app persists conversation state in `storage/session.json` and will restore it on restart.

## Working rule

Trust the instructions above first. Only search further if a needed detail is missing or you discover the instructions are out of date.