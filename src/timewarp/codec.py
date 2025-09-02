from __future__ import annotations

from typing import Any

import orjson
import zstandard as zstd


def to_bytes(obj: Any) -> bytes:
    """Serialize a Python object to canonical JSON bytes using orjson.

    - Sorts keys for deterministic hashing.
    - Ensures UTF-8 validity.
    """

    # We rely on orjson to fail on invalid types unless a default is provided.
    # Callers should pre-normalize (e.g., via Pydantic model_dump(mode="json")).
    return orjson.dumps(obj, option=orjson.OPT_SORT_KEYS)


def from_bytes(data: bytes) -> Any:
    """Deserialize JSON bytes to Python object using orjson."""
    return orjson.loads(data)


_ZSTD_LEVEL_DEFAULT = 8


def zstd_compress(data: bytes, *, level: int = _ZSTD_LEVEL_DEFAULT) -> bytes:
    """Compress bytes with Zstandard (one-shot).

    One-shot API is sufficient for typical payload sizes; streaming can be added later
    for extremely large blobs.
    """

    compressor = zstd.ZstdCompressor(level=level)
    return compressor.compress(data)


def zstd_decompress(data: bytes) -> bytes:
    """Decompress Zstandard-compressed bytes (one-shot)."""
    decompressor = zstd.ZstdDecompressor()
    return decompressor.decompress(data)
