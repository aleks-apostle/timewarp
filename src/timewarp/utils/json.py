from __future__ import annotations

from typing import Any

from ..codec import from_bytes, to_bytes


def dumps(obj: Any) -> bytes:
    """Canonical JSON serialization via codec.to_bytes (sorted keys, orjson).

    Returns UTF-8 encoded bytes. Callers can ``.decode("utf-8")`` for text.
    """

    return to_bytes(obj)


def loads(data: bytes) -> Any:
    """Parse JSON bytes using codec.from_bytes (orjson)."""

    return from_bytes(data)
