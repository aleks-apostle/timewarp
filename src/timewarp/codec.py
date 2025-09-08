from __future__ import annotations

import io
import os
from typing import Any, Final

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


def _env_int(name: str, default: int) -> int:
    try:
        v = os.environ.get(name)
        if v is None:
            return default
        return int(v.strip())
    except Exception:
        return default


_ZSTD_LEVEL_DEFAULT: Final[int] = _env_int("TIMEWARP_ZSTD_LEVEL", 8)
_STREAMING_THRESHOLD: Final[int] = _env_int("TIMEWARP_ZSTD_STREAMING_THRESHOLD", 8 << 20)  # bytes


def zstd_compress(data: bytes, *, level: int = _ZSTD_LEVEL_DEFAULT) -> bytes:
    """Compress bytes with Zstandard.

    Uses one-shot compression for small payloads; switches to streaming when size
    exceeds a threshold to avoid large peak memory usage.
    """

    if len(data) < _STREAMING_THRESHOLD:
        compressor = zstd.ZstdCompressor(level=level)
        return compressor.compress(data)
    # Streaming path
    compressor = zstd.ZstdCompressor(level=level)
    out = io.BytesIO()
    with compressor.stream_writer(out, closefd=False) as writer:
        writer.write(data)
        try:
            writer.flush(zstd.FLUSH_FRAME)
        except Exception:
            # best-effort flush
            pass
    return out.getvalue()


def zstd_decompress(data: bytes) -> bytes:
    """Decompress Zstandard-compressed bytes.

    Uses streaming decompression to support frames without a known content size.
    """
    dctx = zstd.ZstdDecompressor()
    with io.BytesIO(data) as src:
        with dctx.stream_reader(src) as reader:
            out = io.BytesIO()
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
            return out.getvalue()
