"""Streamlit chat UI for the agentic book RAG.

The agent (Claude) is built once per session and reused, with a persistent
``Context`` so multi-turn chat keeps its history. Retrieved passages from each
turn are shown in a "Sources" expander.
"""

from __future__ import annotations

import asyncio
import json

import streamlit as st
from llama_index.core.workflow import Context

from rag_book import config
from rag_book.agent import build_agent, run_streaming
from rag_book.ingest import ingest

st.set_page_config(page_title="Book RAG", page_icon="📚", layout="wide")


def _indexed_books() -> list[str]:
    if config.MANIFEST_PATH.exists():
        return sorted(json.loads(config.MANIFEST_PATH.read_text()).keys())
    return []


def _ensure_agent() -> None:
    if "bundle" not in st.session_state:
        bundle = build_agent()
        st.session_state.bundle = bundle
        st.session_state.ctx = Context(bundle.agent)
        st.session_state.messages = []


def _fmt_tool(kind: str, name: str, payload: object) -> str:
    """Format one agent step (a tool call or its result) for the debug trace."""
    if kind == "call":
        return f"🔧 **call** `{name}` · args={payload}"
    text = str(payload).replace("\n", " ")
    if len(text) > 300:
        text = text[:300] + "…"
    return f"↳ **result** `{name}` → {text}"


# --- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.header("📚 Library")
    books = _indexed_books()
    if books:
        st.caption(f"{len(books)} book(s) indexed")
        for book in books:
            st.write(f"• {book}")
    else:
        st.info("No books indexed yet. Add PDFs to `books/` and re-index.")

    if st.button("🔄 Re-index books/", use_container_width=True):
        with st.spinner("Ingesting PDFs (local embeddings, no API calls)…"):
            summary = ingest()
        # Force a fresh agent so it sees the new index.
        for key in ("bundle", "ctx", "messages"):
            st.session_state.pop(key, None)
        st.success(f"Indexed {summary['num_books']} book(s), {summary['num_chunks']} chunk(s).")
        st.rerun()

    st.divider()
    st.toggle("🐞 Debug: show tool calls", key="debug")
    st.caption(f"LLM: `{config.LLM_MODEL}` · Embeddings: `{config.EMBED_MODEL}` (local)")

# --- Main chat -----------------------------------------------------------
st.title("Chat with your books")

if not _indexed_books():
    st.warning("Add PDFs to the `books/` folder, then click **Re-index books/**.")
    st.stop()

_ensure_agent()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("trace"):
            with st.expander(f"🐞 Debug trace ({len(msg['trace'])} steps)"):
                st.markdown("\n\n".join(msg["trace"]))
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['book']}** — {src['chapter']} — p.{src['page']}  \n"
                        f"> {src['snippet']}"
                    )

if prompt := st.chat_input("Ask for a chapter summary or a question about the books…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    debug = st.session_state.get("debug", False)
    bundle = st.session_state.bundle
    bundle.clear_sources()
    trace_lines: list[str] = []
    answer_acc = {"text": ""}

    with st.chat_message("assistant"):
        trace_box = st.empty() if debug else None
        answer_box = st.empty()

        def on_tool(kind: str, name: str, payload: object) -> None:
            trace_lines.append(_fmt_tool(kind, name, payload))
            if trace_box is not None:
                trace_box.markdown("\n\n".join(trace_lines))

        def on_delta(delta: str) -> None:
            answer_acc["text"] += delta
            answer_box.markdown(answer_acc["text"] + "▌")

        with st.spinner(f"Querying {config.LLM_MODEL} via OpenRouter…"):
            answer = asyncio.run(
                run_streaming(bundle, st.session_state.ctx, prompt, on_tool, on_delta)
            )
        answer_box.markdown(answer)

        # Collapse the live trace into an expander once the run finishes.
        if trace_box is not None and trace_lines:
            with (
                trace_box.container(),
                st.expander(f"🐞 Debug trace ({len(trace_lines)} steps)", expanded=True),
            ):
                st.markdown("\n\n".join(trace_lines))

        sources = list(bundle.sources)
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for src in sources:
                    st.markdown(
                        f"**{src['book']}** — {src['chapter']} — p.{src['page']}  \n"
                        f"> {src['snippet']}"
                    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "trace": trace_lines if debug else [],
        }
    )
