from __future__ import annotations

from timewarp.codec import from_bytes, to_bytes, zstd_compress, zstd_decompress


def test_json_roundtrip_sorted_keys() -> None:
    obj = {"b": 2, "a": 1}
    data = to_bytes(obj)
    # Deterministic ordering ensures 'a' appears before 'b'
    assert data == b'{"a":1,"b":2}'
    assert from_bytes(data) == obj


def test_zstd_roundtrip() -> None:
    payload = b"x" * 1024
    comp = zstd_compress(payload)
    assert len(comp) < len(payload)
    assert zstd_decompress(comp) == payload
