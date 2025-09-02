from __future__ import annotations

from timewarp.pruners import messages_pruner


def test_messages_pruner_truncates_and_limits() -> None:
    state = {
        "values": {
            "messages": [
                {"role": "user", "content": "x" * 100},
                {"role": "assistant", "content": "y" * 80},
                {"role": "assistant", "content": "z" * 60},
            ],
            "other": [{"foo": "bar"}],
            "history": ["a", "b", "c"],
        },
        "content": "t" * 200,
    }
    pruner = messages_pruner(max_len=50, max_items=2)
    pruned = pruner(state)

    assert isinstance(pruned, dict)
    msgs = pruned["values"]["messages"]
    assert isinstance(msgs, list)
    assert len(msgs) == 2
    assert all(len(m.get("content", "")) <= 50 for m in msgs)
    # history list is also limited
    assert len(pruned["values"]["history"]) == 2
    # top-level content string truncated
    assert isinstance(pruned["content"], str) and len(pruned["content"]) <= 50
