from __future__ import annotations

from typing import Any

from ..codec import to_bytes
from ..events import hash_bytes


def hash_prompt(*, messages: Any | None, prompt: Any | None) -> str:
    """Compute a stable sha256 hex over LLM prompt/messages.

    - Prefer ``messages`` when provided; otherwise hash ``prompt``.
    - Falls back to ``repr`` envelope when serialization fails.
    """

    try:
        obj: Any
        if messages is not None:
            obj = {"messages": messages}
        else:
            obj = {"prompt": prompt}
        return hash_bytes(to_bytes(obj))
    except Exception:
        return hash_bytes(to_bytes({"_repr": repr(messages if messages is not None else prompt)}))


def hash_tools_list(tools: Any) -> str:
    """Stable hash for a tools list/specs object."""

    try:
        return hash_bytes(to_bytes({"tools": tools}))
    except Exception:
        return hash_bytes(to_bytes({"_repr": repr(tools)}))


def hash_prompt_ctx(*, messages: Any, tools: Any) -> str:
    """Stable hash for prompt context (messages + tools)."""

    try:
        return hash_bytes(to_bytes({"messages": messages, "tools": tools}))
    except Exception:
        return hash_bytes(
            to_bytes(
                {
                    "_repr": {
                        "messages": repr(messages),
                        "tools": repr(tools),
                    }
                }
            )
        )
