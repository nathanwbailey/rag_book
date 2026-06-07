# Book RAG — agentic RAG over your PDF books

Query a personal library of PDF books — chapter summaries, content questions,
cross-book lookups — through an **agentic** RAG system where the LLM decides
how to search the documents and reports the best answer.

- **Agent / LLM:** [OpenRouter](https://openrouter.io) free tier. Default
  `meta-llama/llama-3.3-70b-instruct:free`; set `OPENROUTER_MODEL` to any free,
  tool-calling slug from https://openrouter.ai/models?max_price=0.
- **Embeddings:** local `BAAI/bge-base-en-v1.5` — offline, no extra key.
- **Vector store:** Chroma (persisted under `storage/`).
- **UI:** Streamlit chat with source citations.
- **Tooling:** `uv` + `ruff`.

## Search architecture

Each query runs a three-stage hybrid pipeline:

1. **BM25 retriever** — keyword matching over all indexed chunks
2. **Vector retriever** — dense semantic similarity via the local BGE embedding model
3. **Reciprocal Rank Fusion** — merges both ranked lists using RRF scores
4. **Cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — rescores the fused top-N chunks before passing them to the LLM

The LLM then synthesizes a grounded answer with inline citations.

## LLM-call boundary

The **only** LLM (OpenRouter) calls happen when you ask a question. Ingestion
(PDF parsing, chapter mapping, chunking, embedding) is fully local and makes
**no** LLM calls — you can run it without `OPENROUTER_API_KEY`.

## Prerequisites

- **Python 3.10–3.13**
- **uv** — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`
  (or see https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
uv sync                       # creates .venv and installs all dependencies
cp .env.example .env          # then add your OpenRouter key (querying only)
```

## Use

```bash
# 1. Drop PDF books into ./books/
# 2. Build the index (no API key needed):
uv run python -m scripts.ingest
# 3. Launch the chat UI:
uv run streamlit run app.py
```

> **Important:** always prefix commands with `uv run`. This ensures they use
> the `.venv` Python where all dependencies are installed. Running `python` or
> `streamlit` directly uses your system Python and will fail with
> `No module named 'llama_index...'`.

In the app: ask "Summarize chapter 3 of \<book\>", "What does \<book\> say about
\<topic\>?", or "Which of my books discuss \<theme\>?". Retrieved passages appear
under **Sources** with book / chapter / page.

## How the agent picks tools

| You ask… | Tool the agent calls |
|---|---|
| Summarize chapter N | `read_chapter` (full chapter → summary) |
| A content question | `search_books` (BM25 + vector + rerank → synthesis) |
| What books / chapters do I have | `list_books` / `list_chapters` |

## Config

Knobs live in [rag_book/config.py](rag_book/config.py): `LLM_MODEL` (or
`OPENROUTER_MODEL` env var), `EMBED_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`,
`TOP_K`.

## Project structure

```
app.py                  Streamlit chat UI
rag_book/
  config.py             Paths, model names, lazy model initialisation
  agent.py              FunctionAgent + search_books / read_chapter tools
  index.py              Chroma-backed VectorStoreIndex helpers
  ingest.py             PDF → chunks → embeddings → Chroma
  pdf_loader.py         PyMuPDF loader with chapter detection via PDF TOC
scripts/
  ingest.py             CLI entry point: python -m scripts.ingest
```

## Dev

```bash
uv run ruff check .
uv run ruff format .
```

## Troubleshooting

**`No module named 'llama_index...'` (or similar)**  
You're running with system Python instead of the project venv. Prefix every
command with `uv run`, or activate the venv first with `source .venv/bin/activate`.

## Notes

- Chapter detection uses each PDF's embedded table of contents (bookmarks). A
  PDF without a TOC is indexed as a single "Full text" chapter.
- Re-indexing rebuilds the Chroma collection from scratch (no duplicates).
- Free OpenRouter endpoints are rate-limited and may log/train on inputs.

## License

MIT — see [LICENSE](LICENSE).
