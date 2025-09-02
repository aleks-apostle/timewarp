from __future__ import annotations

from collections.abc import Callable
from typing import Any, Final


def messages_pruner(max_len: int = 2000, max_items: int = 200) -> Callable[[Any], Any]:
    """Return a state pruner that truncates common message-like fields.

    - Limits list sizes for keys that commonly hold messages: "messages", "history",
      "conversation", "msgs".
    - Truncates string content under message-like contexts (e.g., dict key "content") to
      at most ``max_len`` characters.

    The returned callable is suitable for `LangGraphRecorder.state_pruner`.
    It preserves the overall JSON structure (dict/list) and only prunes where helpful.
    """

    MSG_KEYS: Final[set[str]] = {"messages", "history", "conversation", "msgs"}

    def _prune(value: Any, *, parent_key: str | None, in_msgs_ctx: bool) -> Any:
        # Strings: truncate when in a messages context or explicitly under a content key
        if isinstance(value, str):
            if in_msgs_ctx or parent_key == "content":
                return value[:max_len]
            return value
        # Dict: recurse; entering messages context if key is message-like
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for k, v in value.items():
                k_str = str(k)
                next_ctx = in_msgs_ctx or (k_str in MSG_KEYS)
                pruned_v = _prune(v, parent_key=k_str, in_msgs_ctx=next_ctx)
                # Apply list length limits only when the key suggests message lists
                if isinstance(pruned_v, list) and (k_str in MSG_KEYS):
                    pruned_v = pruned_v[:max_items]
                out[k_str] = pruned_v
            return out
        # List: optionally limit items when already inside a messages context
        if isinstance(value, list):
            items = value[: max_items if in_msgs_ctx else len(value)]
            return [_prune(v, parent_key=parent_key, in_msgs_ctx=in_msgs_ctx) for v in items]
        # Other scalars
        return value

    def _entry(obj: Any) -> Any:
        if isinstance(obj, dict | list):
            return _prune(obj, parent_key=None, in_msgs_ctx=False)
        # Keep behavior safe for unexpected types by returning as-is
        return obj

    return _entry
