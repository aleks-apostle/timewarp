from __future__ import annotations

from timewarp.cli.helpers.state import format_state_pretty


def test_format_state_pretty_truncates_and_counts() -> None:
    state = {
        "long": "x" * 300,
        "list": list(range(60)),
        "nested": {"msg": "y" * 210},
    }
    out = format_state_pretty(state, max_str=100, max_items=10)
    # Truncated string hint present
    assert "<truncated" in out
    # List count hint present
    assert "<... 50 more items>" in out
