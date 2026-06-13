"""Agentic RAG: the LLM drives retrieval tools and reports the best answer.

Builds a LlamaIndex ``FunctionAgent`` (function/tool-calling loop) over the
shared ``RagEngine``. The agent decides which tool to call: ``read_chapter`` for
chapter summaries, ``search_books`` for content questions, ``list_books`` /
``list_chapters`` for navigation. This is the opt-in mode; the default path is
the simpler ``engine.run_simple_streaming``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from llama_index.core.agent.workflow import (
    AgentStream,
    FunctionAgent,
    ToolCall,
    ToolCallResult,
)
from llama_index.core.tools import FunctionTool

from .engine import RagEngine

SYSTEM_PROMPT = """\
You are a research assistant for a personal library of books that have been \
indexed for retrieval. Answer ONLY from the books via your tools — never from \
prior knowledge.

When a tool returns passages or a synthesized answer with sources, that IS the \
material to answer from — use it. Do NOT reply that you couldn't find anything \
when a tool just returned relevant content with sources. Only say the library \
lacks the answer when the tools genuinely return nothing relevant.

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


@dataclass
class AgentBundle:
    """The agent plus a per-turn scratchpad of retrieved sources for the UI."""

    agent: FunctionAgent
    sources: list[dict] = field(default_factory=list)

    def clear_sources(self) -> None:
        self.sources.clear()


def build_agent(engine: RagEngine) -> AgentBundle:
    """Construct the tool-calling agent over the shared retrieval engine."""
    bundle = AgentBundle(agent=None)  # type: ignore[arg-type]

    def search_books(query: str, book: str | None = None) -> str:
        """Hybrid search: BM25 + vector, fused with RRF and cross-encoder
        reranked. Optionally restrict to one book by exact title. Returns the
        retrieved passages (citation-tagged) for you to answer from."""
        passages, sources = engine.retrieve_passages(query, book=book)
        bundle.sources.extend(sources)
        return passages

    def list_books() -> str:
        """List the titles of all indexed books."""
        return engine.list_books()

    def list_chapters(book: str) -> str:
        """List the chapters of one book (by exact title)."""
        return engine.list_chapters(book)

    def read_chapter(book: str, chapter: str) -> str:
        """Return the full text of one chapter (exact book + chapter titles) so
        you can summarize it. Use this for chapter-summary requests."""
        text, sources = engine.read_chapter(book, chapter)
        bundle.sources.extend(sources)
        return text

    bundle.agent = FunctionAgent(
        tools=[
            FunctionTool.from_defaults(fn=search_books),
            FunctionTool.from_defaults(fn=list_books),
            FunctionTool.from_defaults(fn=list_chapters),
            FunctionTool.from_defaults(fn=read_chapter),
        ],
        llm=engine.llm,
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
    ``"call"`` (payload = kwargs dict), ``"result"`` (payload = output text), or
    ``"error"`` (payload = exception detail). ``on_delta(text)`` fires for each
    streamed token of the model's output.
    """
    handler = bundle.agent.run(prompt, ctx=ctx)
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            if on_tool is not None:
                out = event.tool_output
                if getattr(out, "is_error", False):
                    # The agent swallows tool exceptions into an error ToolOutput
                    # (content = str(exc), often empty). Surface it so a failing
                    # tool shows up in the trace instead of a blank result line.
                    exc = getattr(out, "exception", None)
                    detail = str(out.content).strip() or (repr(exc) if exc else "(no detail)")
                    on_tool("error", event.tool_name, detail)
                else:
                    output = getattr(out, "content", out)
                    on_tool("result", event.tool_name, str(output))
        elif isinstance(event, ToolCall):
            if on_tool is not None:
                on_tool("call", event.tool_name, dict(event.tool_kwargs))
        elif isinstance(event, AgentStream):
            if on_delta is not None and event.delta:
                on_delta(event.delta)
    return str(await handler)
