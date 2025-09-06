from __future__ import annotations

from timewarp.codec import zstd_compress, zstd_decompress


def test_zstd_streaming_large_payload_roundtrip() -> None:
    # 10 MiB payload to exceed streaming threshold (8 MiB)
    size = 10 * (1 << 20)
    data = (b"abc123XYZ\n" * (size // 10))[:size]
    comp = zstd_compress(data)
    assert isinstance(comp, bytes | bytearray)
    # Should compress to smaller size for repetitive content
    assert len(comp) < len(data)
    decomp = zstd_decompress(comp)
    assert decomp == data
