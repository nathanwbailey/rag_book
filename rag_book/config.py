"""Central configuration and lazy model wiring.

Importing this module is cheap and side-effect-free apart from loading ``.env``.
Heavy objects (the local embedding model, the OpenRouter LLM) are created lazily
via ``configure_embeddings()`` / ``get_llm()`` so that the ingestion path never
imports or instantiates the LLM client.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = ROOT / "books"
STORAGE_DIR = ROOT / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
MANIFEST_PATH = STORAGE_DIR / "chapters_manifest.json"
# Cross-reference graph (node_id -> referenced node_ids), built at ingest time.
REFERENCE_GRAPH_PATH = STORAGE_DIR / "reference_graph.json"
# Persisted chat session (transcript + serialized agent Context).
SESSION_PATH = STORAGE_DIR / "session.json"
COLLECTION_NAME = "books"

# --- Models --------------------------------------------------------------
# The LLM (via OpenRouter) handles ALL generation: agent reasoning, synthesis,
# summaries. Default is a FREE OpenRouter model that supports tool calling
# (required by the agent). Override with OPENROUTER_MODEL to any free slug from
# https://openrouter.ai/models?max_price=0 that lists "tools" support, e.g.
# qwen/qwen3-coder:free, openai/gpt-oss-120b:free, z-ai/glm-4.5-air:free.
# Note: free endpoints are rate-limited and lower quality than paid models.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
LLM_MAX_TOKENS = 4096
LLM_CONTEXT_WINDOW = 131_072
# Local embedding model — runs on-device, makes no API calls.
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# --- Chunking / retrieval ------------------------------------------------
CHUNK_SIZE = 768
CHUNK_OVERLAP = 128
TOP_K = 6

_embed_configured = False


def configure_embeddings():
    """Set the global LlamaIndex embed model to the local BGE model.

    Pinning ``Settings.embed_model`` (and never leaving it unset) stops
    LlamaIndex from silently falling back to OpenAI embeddings. Makes no
    network calls beyond a one-time model download on first run.
    """
    global _embed_configured
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    if not _embed_configured:
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        _embed_configured = True
    return Settings.embed_model


def get_llm():
    """Return the OpenRouter-backed LLM. Only called on the query path.

    Uses the OpenAI-compatible client pointed at OpenRouter, with function
    calling enabled so the agent's tool-use loop works.
    """
    from llama_index.llms.openai_like import OpenAILike

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. It is required for querying "
            "(the agent calls the LLM via OpenRouter), but NOT for ingestion."
        )
    return OpenAILike(
        model=LLM_MODEL,
        api_base=OPENROUTER_BASE_URL,
        api_key=api_key,
        is_chat_model=True,
        is_function_calling_model=True,
        max_tokens=LLM_MAX_TOKENS,
        context_window=LLM_CONTEXT_WINDOW,
    )
