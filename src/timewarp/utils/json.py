from __future__ import annotations

from typing import Any

import orjson


def dumps(obj: Any) -> bytes:
    """Canonical JSON serialization using orjson with sorted keys.

    Returns UTF-8 encoded bytes. Callers can ``.decode("utf-8")`` for text.
    """

    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)


def loads(data: bytes) -> Any:
    """Parse JSON bytes into Python objects using orjson."""

    return orjson.loads(data)
