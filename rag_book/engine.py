"""Shared retrieval engine for both the simple and agentic RAG modes.

``RagEngine`` owns the expensive, stateful retrieval pieces (BM25, the vector
index, the cross-encoder reranker, the one-hop reference expander) and exposes
them as plain methods. The *agentic* mode (``agent.py``) wraps these methods as
LLM tools; the *simple* mode (``run_simple_streaming``) drives them directly:
it infers a structured book/chapter filter from the user's question, runs the
hybrid search under that filter, and synthesizes a grounded answer — no
tool-calling loop.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass

from llama_index.core.bridge.pydantic import ConfigDict
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.retrievers.bm25 import BM25Retriever

from . import config
from .index import get_all_nodes, get_chapter_chunks, load_index, load_reference_graph

# Cap a chapter dump so a huge chapter can't blow past sane limits.
_MAX_CHAPTER_CHARS = 60_000

# How many retrieved passages to fold into the synthesis context.
_SYNTH_TOP_K = 20

SYNTHESIS_PROMPT = """\
You are a research assistant for a personal library of indexed books. Answer the \
reader's question using ONLY the passages provided below — never from prior \
knowledge. If the passages do not contain the answer, say so plainly.

Cite sources inline as (Book — Chapter — p.N). Be concise and grounded.
"""

# Condense a context-dependent follow-up into a standalone search query. Kept
# tiny and output-only so it works on the rate-limited free models too.
_REWRITE_PROMPT = """\
You rewrite a reader's follow-up message into a standalone search query for a \
library of books, using the conversation so far.

Resolve pronouns and back-references ("it", "that", "this", "those", "the \
previous one") to the concrete topic they stand for. Preserve the reader's \
intent and keep the query short; only add the missing context needed to make it \
self-contained. If the message is already self-contained, return it unchanged.

Output ONLY the rewritten query — no quotes, no label, no explanation.
"""


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric word tokens (drops punctuation)."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _contains_phrase(haystack: list[str], needle: list[str]) -> bool:
    """True if ``needle`` appears as a contiguous run of tokens in ``haystack``."""
    if not needle or len(needle) > len(haystack):
        return False
    return any(
        haystack[i : i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1)
    )


def _match_title(query_tokens: list[str], candidates: list[str]) -> str | None:
    """Return the candidate whose title is quoted word-for-word in the query.

    A title matches only when its name appears verbatim as a contiguous phrase
    in the query — we match on the part before any ``:`` subtitle (the
    distinctive anchor, e.g. "Addicted to Anxiety" or "SECTION 2"). When several
    titles match, the longest (most specific) one wins. No fuzzy matching.
    """
    best: str | None = None
    best_len = 0
    for cand in candidates:
        name = _tokenize(cand.split(":", 1)[0])
        if len(name) > best_len and _contains_phrase(query_tokens, name):
            best, best_len = cand, len(name)
    return best


class ReferenceExpander(BaseNodePostprocessor):
    """Follow the cross-reference graph one hop after reranking.

    For each reranked chunk, pull in the chunks it references ("see Chapter 5",
    "p. 42") so the synthesizer can use them as extra context. Added nodes are
    tagged with ``via_reference`` and capped to protect the context window.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph: dict
    node_lookup: dict
    max_added: int = 3

    @classmethod
    def class_name(cls) -> str:
        return "ReferenceExpander"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        if not self.graph or self.max_added <= 0 or not nodes:
            return nodes

        present = {nws.node.node_id for nws in nodes}
        floor = min((nws.score or 0.0) for nws in nodes)
        added: list[NodeWithScore] = []
        for nws in nodes:
            if len(added) >= self.max_added:
                break
            src = nws.node
            for edge in self.graph.get(src.node_id, []):
                if len(added) >= self.max_added:
                    break
                target_id = edge["node_id"]
                if target_id in present:
                    continue
                target = self.node_lookup.get(target_id)
                if target is None:
                    continue
                present.add(target_id)
                src_page = src.metadata.get("page", "?")
                if edge["ref_type"] == "related":
                    via = f"related to {src.metadata.get('book_title', '?')} p.{src_page}"
                else:
                    via = f"p.{src_page} → {edge['ref_type']} {edge['ref_value']}"
                meta = dict(target.metadata)
                meta["via_reference"] = via
                node = TextNode(id_=target.node_id, text=target.text, metadata=meta)
                added.append(NodeWithScore(node=node, score=floor - 0.01))
        return nodes + added


def _load_manifest() -> dict[str, list[str]]:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text())
    return {}


@dataclass
class RagEngine:
    """Shared retrieval over the indexed library, used by both modes."""

    llm: object
    index: object
    manifest: dict[str, list[str]]
    all_nodes: list[TextNode]
    full_bm25: BM25Retriever
    reranker: SentenceTransformerRerank
    expander: ReferenceExpander
    # One-hop reference expansion is on by default; the UI can toggle it.
    expand_references: bool = True
    # Optional per-turn debug sink; emits one line per traversed reference edge.
    on_trace: Callable[[str], None] | None = None

    # --- Retrieval -------------------------------------------------------
    def retrieve(
        self, query: str, book: str | None = None, chapter: str | None = None
    ) -> list[NodeWithScore]:
        """Hybrid BM25 + vector search (RRF fused), reranked, then optionally
        expanded one hop along the reference graph. ``book``/``chapter`` apply a
        structured metadata filter to both retrievers."""
        # Restrict both retrievers when scoped. BM25 has no metadata filter, so
        # build a scoped BM25 from the matching nodes.
        filter_items: list[MetadataFilter] = []
        if book:
            filter_items.append(MetadataFilter(key="book_title", value=book))
        if chapter:
            filter_items.append(MetadataFilter(key="chapter", value=chapter))

        if filter_items:
            scoped = [
                n
                for n in self.all_nodes
                if (not book or n.metadata.get("book_title") == book)
                and (not chapter or n.metadata.get("chapter") == chapter)
            ]
            bm25 = BM25Retriever.from_defaults(
                nodes=scoped or self.all_nodes, similarity_top_k=_SYNTH_TOP_K
            )
            filters = MetadataFilters(filters=filter_items)
        else:
            bm25 = self.full_bm25
            filters = None

        vector = self.index.as_retriever(similarity_top_k=_SYNTH_TOP_K, filters=filters)
        fusion = QueryFusionRetriever(
            retrievers=[bm25, vector],
            llm=self.llm,  # else it resolves Settings.llm and falls back to OpenAI
            similarity_top_k=_SYNTH_TOP_K,
            num_queries=1,  # no LLM query expansion (llm not actually called)
            mode="reciprocal_rerank",  # RRF fusion
            use_async=False,  # avoid a nested event loop inside the agent run
        )

        query_bundle = QueryBundle(query_str=query)
        nodes = fusion.retrieve(query_bundle)
        nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
        if self.expand_references:
            before = {n.node.node_id for n in nodes}
            nodes = self.expander.postprocess_nodes(nodes, query_bundle)
            self._trace_expansion(nodes, before)
        return nodes

    def _trace_expansion(self, nodes: list[NodeWithScore], before: set[str]) -> None:
        """Report each followed reference edge (source → target) to ``on_trace``."""
        if self.on_trace is None:
            return
        added = [
            n for n in nodes if n.node.node_id not in before and n.metadata.get("via_reference")
        ]
        if not added:
            self.on_trace("graph: no references to traverse among top passages")
            return
        self.on_trace(f"graph: traversed {len(added)} reference edge(s)")
        for n in added:
            m = n.metadata
            self.on_trace(
                f"graph: {m['via_reference']} ⇒ {m.get('book_title')} — "
                f"{m.get('chapter')} — p.{m.get('page')}"
            )

    # --- Synthesis -------------------------------------------------------
    def synthesize(
        self,
        query: str,
        nodes: list[NodeWithScore],
        history: list[dict] | None = None,
        on_delta: Callable[[str], None] | None = None,
    ) -> str:
        """Synthesize a grounded answer from retrieved nodes. Streams deltas to
        ``on_delta`` when given; otherwise returns the answer in one shot.
        ``history`` is the visible transcript (for multi-turn chat memory)."""
        messages = self._build_messages(query, nodes, history)
        if on_delta is not None:
            text = ""
            for chunk in self.llm.stream_chat(messages):
                delta = chunk.delta or ""
                if delta:
                    text += delta
                    on_delta(delta)
            return text
        response = self.llm.chat(messages)
        return str(response.message.content or "").strip()

    def _build_messages(
        self, query: str, nodes: list[NodeWithScore], history: list[dict] | None
    ) -> list[ChatMessage]:
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=SYNTHESIS_PROMPT)]
        for msg in history or []:
            role = MessageRole.USER if msg.get("role") == "user" else MessageRole.ASSISTANT
            messages.append(ChatMessage(role=role, content=msg.get("content", "")))
        context = self._format_context(nodes)
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=f"Context passages:\n{context}\n\nQuestion: {query}",
            )
        )
        return messages

    @staticmethod
    def _format_context(nodes: list[NodeWithScore]) -> str:
        if not nodes:
            return "(no passages retrieved)"
        lines = []
        for node in nodes:
            meta = node.metadata
            via = f" [↪ via {meta['via_reference']}]" if meta.get("via_reference") else ""
            lines.append(
                f"[{meta.get('book_title')} — {meta.get('chapter')} — "
                f"p.{meta.get('page')}{via}]\n{node.get_content().strip()}"
            )
        return "\n\n".join(lines)

    # --- Query rewriting -------------------------------------------------
    def rewrite_query(self, query: str, history: list[dict] | None = None) -> str:
        """Rewrite a follow-up into a standalone query using recent turns.

        No-ops (returns ``query`` unchanged) when there is no prior context, so
        the first turn costs no extra LLM call. Any failure degrades to the
        original query.
        """
        recent = [m for m in (history or []) if m.get("content")]
        if not recent:
            return query

        convo = "\n".join(f"{m.get('role')}: {m.get('content', '')}" for m in recent[-4:])
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_REWRITE_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=f"Conversation:\n{convo}\n\nFollow-up: {query}",
            ),
        ]
        try:
            raw = str(self.llm.chat(messages).message.content or "")
        except Exception:
            return query

        # Take the first non-empty line, strip quotes / a stray "query:" label.
        rewritten = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
        rewritten = rewritten.strip("\"'").strip()
        if ":" in rewritten[:20]:
            label, _, rest = rewritten.partition(":")
            if "query" in label.lower():
                rewritten = rest.strip()
        # Guard against a model that ignored the instruction and answered instead.
        if not rewritten or len(rewritten) > 300:
            return query
        return rewritten

    # --- Structured-data inference --------------------------------------
    def infer_filters(self, query: str) -> tuple[str | None, str | None]:
        """Infer a (book, chapter) filter from the query text alone — no LLM.

        A book/chapter is only selected when its title is quoted word-for-word in
        the query (see ``_match_title``). Returns ``(None, None)`` when nothing
        matches → search the whole library.
        """
        if not self.manifest:
            return None, None

        query_tokens = _tokenize(query)
        book = _match_title(query_tokens, list(self.manifest))

        # A chapter is only meaningful once we know which book it belongs to.
        chapter = _match_title(query_tokens, self.manifest.get(book, [])) if book else None
        return book, chapter

    # --- Convenience used by the agent tools ----------------------------
    def search(
        self, query: str, book: str | None = None, chapter: str | None = None
    ) -> tuple[str, list[dict]]:
        """Retrieve + synthesize (non-streaming). Returns (answer, sources)."""
        nodes = self.retrieve(query, book=book, chapter=chapter)
        answer = self.synthesize(query, nodes)
        return answer, self.nodes_to_sources(nodes)

    def read_chapter(self, book: str, chapter: str) -> tuple[str, list[dict]]:
        """Return the full text of one chapter (exact titles), page-ordered."""
        pairs = get_chapter_chunks(book, chapter)
        if not pairs:
            return (
                f"No chapter '{chapter}' found in '{book}'. "
                "Call list_chapters to see exact titles.",
                [],
            )
        out: list[str] = []
        sources: list[dict] = []
        total = 0
        for meta, text in pairs:
            sources.append(self._source(meta, text.strip().replace("\n", " ")[:200]))
            piece = f"[p.{meta.get('page')}] {text.strip()}"
            if total + len(piece) > _MAX_CHAPTER_CHARS:
                out.append("...[chapter truncated]...")
                break
            out.append(piece)
            total += len(piece)
        return f"{book} — {chapter}\n\n" + "\n\n".join(out), sources

    def list_books(self) -> str:
        """List the titles of all indexed books."""
        if not self.manifest:
            return "No books are indexed yet."
        return "\n".join(f"- {book}" for book in sorted(self.manifest))

    def list_chapters(self, book: str) -> str:
        """List the chapters of one book (by exact title)."""
        chapters = self.manifest.get(book)
        if chapters is None:
            available = ", ".join(sorted(self.manifest)) or "none"
            return f"No book titled '{book}'. Indexed books: {available}."
        return "\n".join(f"{i + 1}. {c}" for i, c in enumerate(chapters))

    # --- Sources ---------------------------------------------------------
    def nodes_to_sources(self, nodes: list[NodeWithScore]) -> list[dict]:
        return [
            self._source(n.metadata, n.get_content().strip().replace("\n", " ")[:200])
            for n in nodes
        ]

    @staticmethod
    def _source(meta: dict, snippet: str) -> dict:
        return {
            "book": meta.get("book_title", "?"),
            "chapter": meta.get("chapter", "?"),
            "page": meta.get("page", "?"),
            "snippet": snippet,
            "via_reference": meta.get("via_reference"),
        }


def build_engine() -> RagEngine:
    """Construct the shared retrieval engine over the persisted index (once)."""
    config.configure_embeddings()
    llm = config.get_llm()
    index = load_index()

    # Build the expensive / stateful retrieval pieces ONCE, not per query. BM25
    # is built from the actual Chroma nodes (index.docstore is empty for a
    # Chroma-backed index). The cross-encoder reranker loads its model here too.
    all_nodes = get_all_nodes()
    full_bm25 = BM25Retriever.from_defaults(nodes=all_nodes, similarity_top_k=_SYNTH_TOP_K)
    reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=5)

    # One-hop reference expansion: the graph + a node_id -> TextNode lookup (reuse
    # the nodes already loaded for BM25 rather than refetching from Chroma).
    node_lookup = {node.node_id: node for node in all_nodes}
    expander = ReferenceExpander(graph=load_reference_graph(), node_lookup=node_lookup)

    return RagEngine(
        llm=llm,
        index=index,
        manifest=_load_manifest(),
        all_nodes=all_nodes,
        full_bm25=full_bm25,
        reranker=reranker,
        expander=expander,
    )


async def run_simple_streaming(
    engine: RagEngine,
    query: str,
    history: list[dict] | None = None,
    on_delta: Callable[[str], None] | None = None,
    on_info: Callable[[str], None] | None = None,
) -> tuple[str, list[dict]]:
    """Simple (non-agentic) RAG: infer a structured filter, retrieve under it,
    then synthesize a grounded answer with chat memory.

    Returns (answer, sources). ``on_delta`` streams answer tokens; ``on_info``
    surfaces the rewritten query and inferred filter for the debug trace.
    """
    # Resolve context-dependent follow-ups ("how can I reduce it?") into a
    # standalone query before anything downstream sees it.
    search_query = engine.rewrite_query(query, history)
    if on_info is not None and search_query != query:
        on_info(f"rewrote query → {search_query}")

    book, chapter = engine.infer_filters(search_query)
    if on_info is not None:
        parts = []
        if book:
            parts.append(f"book={book}")
        if chapter:
            parts.append(f"chapter={chapter}")
        on_info(f"inferred filter: {', '.join(parts) or 'none (whole library)'}")

    nodes = engine.retrieve(search_query, book=book, chapter=chapter)
    if on_info is not None:
        expanded = sum(1 for n in nodes if n.metadata.get("via_reference"))
        extra = f" (+{expanded} via references)" if expanded else ""
        scope = "filtered" if (book or chapter) else "whole library"
        on_info(f"hybrid search ({scope}) → {len(nodes)} passage(s){extra}")
        for n in nodes:
            m = n.metadata
            via = f" [↪ {m['via_reference']}]" if m.get("via_reference") else ""
            on_info(f"· {m.get('book_title')} — {m.get('chapter')} — p.{m.get('page')}{via}")
        on_info("synthesizing answer…")

    answer = engine.synthesize(search_query, nodes, history=history, on_delta=on_delta)
    return answer, engine.nodes_to_sources(nodes)
