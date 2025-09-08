from __future__ import annotations

from timewarp.codec import from_bytes
from timewarp.langgraph.serialize import normalize_bytes


def test_normalize_bytes_applies_privacy_marks() -> None:
    obj = {
        "user": {"name": "Alice", "ssn": "123-45-6789"},
        "api_key": "SECRET-XYZ",
        "nested": {"list": [1, 2, {"secret": "HIDE"}]},
    }
    marks = {"user.ssn": "mask4", "api_key": "redact", "nested.list[2].secret": "redact"}
    b = normalize_bytes(obj, privacy_marks=marks)
    out = from_bytes(b)
    assert out["user"]["ssn"].endswith("6789") and out["user"]["ssn"].startswith("***")
    assert out["api_key"] == "[REDACTED]"
    assert out["nested"]["list"][2]["secret"] == "[REDACTED]"
