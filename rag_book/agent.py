"""Agentic RAG: the LLM drives retrieval tools and reports the best answer.

Builds a LlamaIndex ``FunctionAgent`` (function/tool-calling loop) over the
indexed library, backed by an OpenRouter model. The agent decides which tool to
call: ``read_chapter`` for chapter summaries, ``search_books`` for content
questions, ``list_books`` / ``list_chapters`` for navigation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from llama_index.core.agent.workflow import (
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.tools import FunctionTool
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever

from . import config
from .index import get_all_nodes, get_chapter_chunks, load_index

SYSTEM_PROMPT = """\
You are a research assistant for a personal library of books that have been \
indexed for retrieval. Answer ONLY from the books via your tools — never from \
prior knowledge. If the tools do not contain the answer, say so plainly.

Tool guidance:
- "Summarize chapter X of <book>" or any chapter-level request -> use \
read_chapter to pull the full chapter, then summarize it.
- A factual/content question ("what does the book say about ...") -> use \
search_books. Run additional searches with refined queries if the first \
results are thin.
- "Which books ...", "what do I have" -> use list_books / list_chapters.

Always ground your answer in what the tools returned and cite sources inline as \
(Book — Chapter — p.N). When unsure which book the user means, call list_books \
first.
"""

# Cap a chapter dump so a huge chapter can't blow past sane limits.
_MAX_CHAPTER_CHARS = 60_000


@dataclass
class AgentBundle:
    """The agent plus a per-turn scratchpad of retrieved sources for the UI."""

    agent: FunctionAgent
    sources: list[dict] = field(default_factory=list)

    def clear_sources(self) -> None:
        self.sources.clear()


def _load_manifest() -> dict[str, list[str]]:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text())
    return {}


def build_agent() -> AgentBundle:
    """Construct the agent and its tools over the persisted index."""
    config.configure_embeddings()
    llm = config.get_llm()
    index = load_index()
    manifest = _load_manifest()
    bundle = AgentBundle(agent=None)  # type: ignore[arg-type]

    # Build the expensive / stateful retrieval pieces ONCE, not per query.
    # BM25 is built from the actual Chroma nodes (index.docstore is empty for a
    # Chroma-backed index). The cross-encoder reranker loads its model here too.
    all_nodes = get_all_nodes()
    full_bm25 = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=20)
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5)

    def _record(meta: dict, snippet: str = "") -> None:
        bundle.sources.append(
            {
                "book": meta.get("book_title", "?"),
                "chapter": meta.get("chapter", "?"),
                "page": meta.get("page", "?"),
                "snippet": snippet,
            }
        )

    def search_books(query: str, book: str | None = None) -> str:
        """Hybrid search: BM25 + vector, fused with RRF and cross-encoder
        reranked. Optionally restrict to one book by exact title. Returns a
        synthesized answer plus the source passages."""
        # Restrict both retrievers to one book when asked. BM25 has no metadata
        # filter, so build a per-book BM25 from the matching nodes.
        if book:
            book_nodes = [n for n in all_nodes if n.metadata.get("book_title") == book]
            bm25 = BM25Retriever.from_defaults(nodes=book_nodes or all_nodes, similarity_top_k=20)
            filters = MetadataFilters(filters=[MetadataFilter(key="book_title", value=book)])
        else:
            bm25 = full_bm25
            filters = None

        vector = index.as_retriever(similarity_top_k=20, filters=filters)
        fusion = QueryFusionRetriever(
            retrievers=[bm25, vector],
            llm=llm,  # else it resolves Settings.llm and falls back to OpenAI
            similarity_top_k=20,
            num_queries=1,  # no LLM query expansion (llm not actually called)
            mode="reciprocal_rerank",  # RRF fusion
            use_async=False,  # avoid a nested event loop inside the agent run
        )
        engine = RetrieverQueryEngine.from_args(
            retriever=fusion,
            node_postprocessors=[reranker],
            llm=llm,
        )
        response = engine.query(query)

        lines = [str(response).strip(), "", "Sources:"]
        for node in response.source_nodes:
            meta = node.metadata
            snippet = node.get_content().strip().replace("\n", " ")[:200]
            _record(meta, snippet)
            lines.append(
                f"- {meta.get('book_title')} — {meta.get('chapter')} — "
                f"p.{meta.get('page')}: {snippet}"
            )
        return "\n".join(lines)

    def list_books() -> str:
        """List the titles of all indexed books."""
        if not manifest:
            return "No books are indexed yet."
        return "\n".join(f"- {book}" for book in sorted(manifest))

    def list_chapters(book: str) -> str:
        """List the chapters of one book (by exact title)."""
        chapters = manifest.get(book)
        if chapters is None:
            available = ", ".join(sorted(manifest)) or "none"
            return f"No book titled '{book}'. Indexed books: {available}."
        return "\n".join(f"{i + 1}. {c}" for i, c in enumerate(chapters))

    def read_chapter(book: str, chapter: str) -> str:
        """Return the full text of one chapter (exact book + chapter titles) so
        you can summarize it. Use this for chapter-summary requests."""
        pairs = get_chapter_chunks(book, chapter)
        if not pairs:
            return (
                f"No chapter '{chapter}' found in '{book}'. Call list_chapters to see exact titles."
            )
        out: list[str] = []
        total = 0
        for meta, text in pairs:
            _record(meta, text.strip().replace("\n", " ")[:200])
            piece = f"[p.{meta.get('page')}] {text.strip()}"
            if total + len(piece) > _MAX_CHAPTER_CHARS:
                out.append("...[chapter truncated]...")
                break
            out.append(piece)
            total += len(piece)
        header = f"{book} — {chapter}\n\n"
        return header + "\n\n".join(out)

    bundle.agent = FunctionAgent(
        tools=[
            FunctionTool.from_defaults(fn=search_books),
            FunctionTool.from_defaults(fn=list_books),
            FunctionTool.from_defaults(fn=list_chapters),
            FunctionTool.from_defaults(fn=read_chapter),
        ],
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
    )
    return bundle


async def run_streaming(
    bundle: AgentBundle,
    ctx,
    prompt: str,
    on_tool: Callable[[str, str, object], None] | None = None,
    on_delta: Callable[[str], None] | None = None,
) -> str:
    """Run the agent, surfacing progress via callbacks. Returns the final answer.

    ``on_tool(kind, name, payload)`` fires for each tool step — ``kind`` is
    ``"call"`` (payload = kwargs dict) or ``"result"`` (payload = output text).
    ``on_delta(text)`` fires for each streamed token of the model's output.
    """
    handler = bundle.agent.run(prompt, ctx=ctx)
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            if on_tool is not None:
                output = getattr(event.tool_output, "content", event.tool_output)
                on_tool("result", event.tool_name, str(output))
        elif isinstance(event, ToolCall):
            if on_tool is not None:
                on_tool("call", event.tool_name, dict(event.tool_kwargs))
        elif isinstance(event, AgentStream):
            if on_delta is not None and event.delta:
                on_delta(event.delta)
    return str(await handler)
