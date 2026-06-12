# Book RAG — agentic RAG over your PDF books

Query a personal library of PDF books — chapter summaries, content questions,
cross-book lookups — through an **agentic** RAG system where the LLM decides
how to search the documents and reports the best answer.

- **Agent / LLM:** [OpenRouter](https://openrouter.io) free tier. Default
  `meta-llama/llama-3.3-70b-instruct:free`; set `OPENROUTER_MODEL` to any free,
  tool-calling slug from https://openrouter.ai/models?max_price=0.
- **Embeddings:** local `BAAI/bge-base-en-v1.5` — offline, no extra key.
- **Vector store:** Chroma (persisted under `storage/`).
- **Reference graph:** chunks linked by in-text cross-references ("see Chapter 5",
  "p. 42"), followed one hop at query time for extra context.
- **UI:** Streamlit chat with source citations, persistent conversation, and a
  one-hop toggle.
- **Tooling:** `uv` + `ruff`.

## Search architecture

Each query runs a three-stage hybrid pipeline:

1. **BM25 retriever** — keyword matching over all indexed chunks
2. **Vector retriever** — dense semantic similarity via the local BGE embedding model
3. **Reciprocal Rank Fusion** — merges both ranked lists using RRF scores
4. **Cross-encoder reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`) — rescores the fused top-N chunks before passing them to the LLM

The LLM then synthesizes a grounded answer with inline citations.

### History-aware query rewriting

In simple mode, a context-dependent follow-up is condensed into a standalone
query before retrieval. Ask *"what is chronic pain?"* then *"how can I reduce
it?"* and the second question is rewritten to *"how can I reduce chronic pain"*
so BM25/vector search has the missing subject. The rewrite uses the last few
turns, no-ops on the first turn (no extra LLM call), degrades to the original on
any failure, and is shown in the debug trace as `rewrote query → …`. (Agentic
mode doesn't need this — its tool-calling LLM already resolves context from its
conversation memory.)

### One-hop reference expansion

At ingest time a **cross-reference graph** is built (no LLM): each chunk is scanned
for in-text references like "see Chapter 5" or "as on p. 42" and linked to the
chunks that make up that chapter/page **in the same book**. Chapter references
resolve via the book's printed chapter numbers (parsed from the numbered chapter
titles), so they line up even though our internal numbering also counts front
matter; page references map onto the PDF page and are best-effort. The graph is
saved to `storage/reference_graph.json`.

When **One-hop reference expansion** is enabled (sidebar toggle, on by default),
the reranked top passages are expanded with their referenced neighbours before
synthesis. Passages pulled in this way are flagged in **Sources** with `↪ via …`.
Cross-references are sparse in most prose, so this fires only when a referencing
passage is among the top hits.

### Persistent conversation

The chat transcript **and** the agent's memory are saved to
`storage/session.json` after each turn (the LlamaIndex `Context` is serialized
via `JsonSerializer`), so closing and reopening the app resumes the same
conversation — including follow-ups that rely on earlier turns. Use **Reset
conversation** in the sidebar to clear it. Re-indexing also resets the session,
since chunk ids change on rebuild.

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
  agent.py              FunctionAgent + tools + one-hop ReferenceExpander
  index.py              Chroma-backed VectorStoreIndex + reference-graph helpers
  ingest.py             PDF → chunks → embeddings → Chroma (+ reference graph)
  pdf_loader.py         PyMuPDF loader with chapter detection via PDF TOC
  references.py         Cross-reference extraction + graph builder (no LLM)
  session.py            Persist/restore chat transcript + agent memory
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
