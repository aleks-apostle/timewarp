from __future__ import annotations

from timewarp.codec import zstd_compress, zstd_decompress


def test_zstd_roundtrip_env_knobs(monkeypatch) -> None:
    # Force streaming by setting small threshold
    monkeypatch.setenv("TIMEWARP_ZSTD_STREAMING_THRESHOLD", "64")
    monkeypatch.setenv("TIMEWARP_ZSTD_LEVEL", "5")
    # Prepare payload larger than threshold
    src = (b"0123456789abcdef" * 8) + (b"x" * 64)
    comp = zstd_compress(src)
    out = zstd_decompress(comp)
    assert out == src
