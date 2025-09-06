from __future__ import annotations

from typing import Any

_seen_tags: set[str] = set()


def log_warn_once(
    tag: str, exc: Exception | None = None, attrs: dict[str, Any] | None = None
) -> None:
    """Emit a single-process warning message at most once per tag.

    This avoids flooding logs when optional integrations or best-effort operations fail.
    Uses a simple print fallback to keep logging lightweight and dependency-free.
    """
    global _seen_tags
    if tag in _seen_tags:
        return
    _seen_tags.add(tag)
    parts: list[str] = [f"[timewarp][warn] {tag}"]
    if attrs:
        try:
            parts.append(str({k: v for k, v in attrs.items()}))
        except Exception:
            pass
    if exc is not None:
        try:
            parts.append(f"exc={type(exc).__name__}: {exc}")
        except Exception:
            pass
    try:
        print(" ".join(parts))
    except Exception:
        # Give up quietly
        return
