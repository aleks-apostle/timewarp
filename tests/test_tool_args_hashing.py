from __future__ import annotations

from pathlib import Path
from typing import Any

from timewarp.codec import to_bytes
from timewarp.events import ActionType, Run, hash_bytes
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _FakeUpdatesGraph:
    def __init__(self, updates: list[tuple[str, Any]]) -> None:
        self._updates = updates

    def stream(self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any):
        yield from self._updates


def test_recorder_tool_args_hash(tmp_path: Path) -> None:
    # Prepare one TOOL-like update with explicit args/kwargs
    upd = (
        "updates",
        {"tool_name": "mytool", "args": [1, 2], "kwargs": {"x": 3}},
    )
    graph = _FakeUpdatesGraph([upd])
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="toolargs", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        snapshot_every=0,
        stream_modes=("updates",),
        stream_subgraphs=False,
    )
    _ = rec.invoke({"x": 0}, config={})
    events = store.list_events(run.run_id)
    # Expect a TOOL event with args hash present
    tool_ev = next(e for e in events if e.action_type is ActionType.TOOL)
    assert "args" in tool_ev.hashes
    expected = hash_bytes(to_bytes({"args": [1, 2], "kwargs": {"x": 3}}))
    assert tool_ev.hashes["args"] == expected
