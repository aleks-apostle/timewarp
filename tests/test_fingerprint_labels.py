from __future__ import annotations

from timewarp.utils.fingerprint import runtime_labels


def test_runtime_labels_basic() -> None:
    lab = runtime_labels()
    assert isinstance(lab, dict)
    assert "fp.py" in lab
    assert isinstance(lab["fp.py"], str)
