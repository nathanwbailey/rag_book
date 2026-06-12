"""Persist a chat session (transcript + agent memory) across app restarts.

The visible transcript is a list of plain message dicts. The agent's memory
lives in a LlamaIndex ``Context``, which serializes via ``JsonSerializer``. Both
are stored together in one JSON file so closing and reopening Streamlit resumes
the same conversation. Restoring the Context is best-effort: if the saved shape
no longer matches (e.g. after a library upgrade), we fall back to a fresh
Context rather than failing to start.
"""

from __future__ import annotations

import json

from llama_index.core.workflow import Context, JsonSerializer

from . import config


def save_session(messages: list[dict], ctx: Context | None = None) -> None:
    """Write the transcript and (optionally) the serialized agent Context.

    Simple (non-agentic) mode has no ``Context``; pass ``ctx=None`` to persist
    just the transcript.
    """
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "messages": messages,
        "context": ctx.to_dict(serializer=JsonSerializer()) if ctx is not None else None,
    }
    config.SESSION_PATH.write_text(json.dumps(data))


def load_session(agent=None) -> tuple[list[dict], Context | None] | None:
    """Load the saved transcript and Context, or ``None`` if there is none.

    The transcript is always returned. The ``Context`` is restored only when an
    ``agent`` is given and a serialized context was saved; otherwise the context
    is ``None`` (simple mode, or schema drift on a stored agent context).
    """
    if not config.SESSION_PATH.exists():
        return None
    try:
        data = json.loads(config.SESSION_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    messages = data.get("messages", [])
    raw_ctx = data.get("context")
    if agent is not None and raw_ctx is not None:
        try:
            return messages, Context.from_dict(agent, raw_ctx, serializer=JsonSerializer())
        except Exception:
            pass  # schema drift / corrupt context -> fresh memory, keep transcript
        return messages, Context(agent)
    return messages, None


def reset_session() -> None:
    """Delete the persisted session, if any."""
    config.SESSION_PATH.unlink(missing_ok=True)
