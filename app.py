"""Streamlit chat UI for the book RAG.

Two retrieval modes share one ``RagEngine`` (built once per session):

- **Simple (default):** infer a structured book/chapter filter from the
  question, run the hybrid search under it, synthesize a grounded answer. Chat
  memory is the visible transcript.
- **Agentic (opt-in):** a tool-calling agent drives retrieval, with a persistent
  ``Context`` for memory. Built lazily, only when the mode is enabled.

Retrieved passages from each turn are shown in a "Sources" expander.
"""

from __future__ import annotations

import asyncio
import json

import streamlit as st
from llama_index.core.workflow import Context

from rag_book import config
from rag_book.agent import build_agent, run_streaming
from rag_book.engine import build_engine, run_simple_streaming
from rag_book.ingest import ingest
from rag_book.session import load_session, reset_session, save_session

st.set_page_config(page_title="Book RAG", page_icon="📚", layout="wide")


def _indexed_books() -> list[str]:
    if config.MANIFEST_PATH.exists():
        return sorted(json.loads(config.MANIFEST_PATH.read_text()).keys())
    return []


def _ensure_engine() -> None:
    """Build the shared retrieval engine once and restore the transcript."""
    if "engine" not in st.session_state:
        st.session_state.engine = build_engine()
        restored = load_session()  # transcript only; no agent Context needed
        st.session_state.messages = restored[0] if restored is not None else []


def _ensure_agent() -> None:
    """Build the tool-calling agent + Context lazily (agentic mode only)."""
    if "bundle" not in st.session_state:
        bundle = build_agent(st.session_state.engine)
        st.session_state.bundle = bundle
        # Resume the saved agent memory if one matches; else fresh Context.
        restored = load_session(bundle.agent)
        if restored is not None and restored[1] is not None:
            st.session_state.ctx = restored[1]
        else:
            st.session_state.ctx = Context(bundle.agent)


def _render_source(src: dict) -> None:
    """Render one cited passage, flagging ones pulled in via a reference link."""
    via = src.get("via_reference")
    tag = f" · ↪ via {via}" if via else ""
    st.markdown(
        f"**{src['book']}** — {src['chapter']} — p.{src['page']}{tag}  \n> {src['snippet']}"
    )


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
        # Node ids change on rebuild, so the saved session is stale — drop it.
        reset_session()
        # Force a fresh engine/agent so they see the new index.
        for key in ("engine", "bundle", "ctx", "messages"):
            st.session_state.pop(key, None)
        st.success(
            f"Indexed {summary['num_books']} book(s), {summary['num_chunks']} chunk(s), "
            f"{summary['num_edges']} reference link(s)."
        )
        st.rerun()

    if st.button("🗑️ Reset conversation", use_container_width=True):
        reset_session()
        st.session_state.messages = []
        # Drop the agent memory so it rebuilds fresh next agentic turn.
        bundle = st.session_state.get("bundle")
        if bundle is not None:
            st.session_state.ctx = Context(bundle.agent)
        st.rerun()

    st.divider()
    st.toggle("🤖 Agentic mode", key="agentic", value=False)
    st.toggle("🔗 One-hop reference expansion", key="expand_refs", value=True)
    st.toggle("🐞 Debug: show retrieval steps", key="debug")
    st.caption(f"LLM: `{config.LLM_MODEL}` · Embeddings: `{config.EMBED_MODEL}` (local)")

# --- Main chat -----------------------------------------------------------
st.title("Chat with your books")

if not _indexed_books():
    st.warning("Add PDFs to the `books/` folder, then click **Re-index books/**.")
    st.stop()

_ensure_engine()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("trace"):
            with st.expander(f"🐞 Debug trace ({len(msg['trace'])} steps)"):
                st.markdown("\n\n".join(msg["trace"]))
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})"):
                for src in msg["sources"]:
                    _render_source(src)

if prompt := st.chat_input("Ask for a chapter summary or a question about the books…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agentic = st.session_state.get("agentic", False)
    debug = st.session_state.get("debug", False)
    engine = st.session_state.engine
    engine.expand_references = st.session_state.get("expand_refs", True)
    trace_lines: list[str] = []
    answer_acc = {"text": ""}

    with st.chat_message("assistant"):
        trace_box = st.empty() if debug else None
        answer_box = st.empty()

        def on_tool(kind: str, name: str, payload: object) -> None:
            trace_lines.append(_fmt_tool(kind, name, payload))
            if trace_box is not None:
                trace_box.markdown("\n\n".join(trace_lines))

        def on_info(info: str) -> None:
            trace_lines.append(f"🔎 {info}")
            if trace_box is not None:
                trace_box.markdown("\n\n".join(trace_lines))

        def on_trace(msg: str) -> None:
            trace_lines.append(f"🔗 {msg}")
            if trace_box is not None:
                trace_box.markdown("\n\n".join(trace_lines))

        def on_delta(delta: str) -> None:
            answer_acc["text"] += delta
            answer_box.markdown(answer_acc["text"] + "▌")

        # Surface graph traversal (one-hop expansion) in the debug trace, in
        # both modes — the engine drives retrieval for each.
        engine.on_trace = on_trace if debug else None
        with st.spinner(f"Querying {config.LLM_MODEL} via OpenRouter…"):
            if agentic:
                _ensure_agent()
                bundle = st.session_state.bundle
                bundle.clear_sources()
                answer = asyncio.run(
                    run_streaming(bundle, st.session_state.ctx, prompt, on_tool, on_delta)
                )
                sources = list(bundle.sources)
                ctx_to_save = st.session_state.ctx
            else:
                # History excludes the just-appended current turn (sent separately
                # with its retrieved context by the synthesizer).
                history = st.session_state.messages[:-1]
                answer, sources = asyncio.run(
                    run_simple_streaming(engine, prompt, history, on_delta, on_info)
                )
                ctx_to_save = None
        engine.on_trace = None  # drop the per-turn closure
        answer_box.markdown(answer)

        # Collapse the live trace into an expander once the run finishes.
        if trace_box is not None and trace_lines:
            with (
                trace_box.container(),
                st.expander(f"🐞 Debug trace ({len(trace_lines)} steps)", expanded=True),
            ):
                st.markdown("\n\n".join(trace_lines))

        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for src in sources:
                    _render_source(src)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "trace": trace_lines if debug else [],
        }
    )
    # Persist the transcript (+ agent memory in agentic mode) so the chat
    # survives a restart.
    save_session(st.session_state.messages, ctx_to_save)
