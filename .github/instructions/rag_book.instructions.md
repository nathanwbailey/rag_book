---
applyTo: "rag_book/**/*.py"
---

# rag_book Directory Instructions

This package holds the retrieval engine, ingest pipeline, PDF loader, persistence helpers, and the agent wrapper. It is the core application code for the Streamlit book-RAG app.

## Files in this directory

- `__init__.py`: Package marker.
  - I/O: none.
- `agent.py`: Builds the tool-calling `FunctionAgent` and wraps engine methods as tools.
  - I/O: input is a `RagEngine`; output is an `AgentBundle` plus async streaming helpers that return the final answer.
- `config.py`: Central configuration, path constants, and lazy model setup.
  - I/O: reads environment variables and `.env`; outputs configured embedding and LLM client objects plus repository paths.
- `engine.py`: Shared retrieval, reranking, reference expansion, query rewriting, and answer synthesis.
  - I/O: input is a query, optional book/chapter filters, and chat history; output is retrieved passages, sources, or a synthesized answer.
- `index.py`: Chroma persistence, index loading, node retrieval, and reference-graph storage.
  - I/O: reads and writes the persistent Chroma store and JSON graph files under `storage/`.
- `ingest.py`: End-to-end PDF ingest and index rebuild pipeline.
  - I/O: reads PDFs from `books/`; writes the manifest, reference graph, and Chroma index under `storage/`; returns a build summary.
- `pdf_loader.py`: Reads PDFs with PyMuPDF, detects TOC/chapter structure, and chunks text into `TextNode`s.
  - I/O: reads PDF files from disk; outputs ordered `TextNode` chunks with metadata.
- `references.py`: Builds sequential, chapter-reference, and semantic-bridge edges between chunks.
  - I/O: input is chunk nodes plus embeddings; output is an adjacency map of reference edges.
- `session.py`: Saves and restores chat transcript and agent context.
  - I/O: reads and writes `storage/session.json`; input is message history and optional LlamaIndex `Context`; output is the restored session state.