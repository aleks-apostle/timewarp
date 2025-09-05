from __future__ import annotations

import difflib
from typing import Any


def to_text(obj: Any) -> str:
    if obj is None:
        return ""
    if isinstance(obj, (dict, list)):  # noqa: UP038
        try:
            from ..codec import to_bytes as _to_bytes

            return _to_bytes(obj).decode("utf-8")
        except Exception:
            return str(obj)
    return str(obj)


def struct_diff(a: Any, b: Any) -> dict[str, Any]:
    """Return a DeepDiff dict for comparable JSON-like structures.

    Only attempts structural diff when both inputs are dict or list; otherwise returns {}.
    """
    if not (isinstance(a, (dict, list)) and isinstance(b, (dict, list))):  # noqa: UP038
        return {}
    try:
        from deepdiff import DeepDiff

        dd = DeepDiff(a, b, ignore_order=False)
        return dict(dd)
    except Exception:
        return {}


def text_unified(a_text: str, b_text: str) -> str:
    """Unified text diff between two strings."""
    return "\n".join(difflib.unified_diff(a_text.splitlines(), b_text.splitlines(), lineterm=""))
