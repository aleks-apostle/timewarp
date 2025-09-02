from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from timewarp.langgraph import wrap
from timewarp.store import LocalStore


class _FakeGraph:
    def __init__(self, updates: list[tuple[str, Any]]) -> None:
        self._updates = updates

    def stream(
        self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any
    ) -> Iterable[Any]:
        yield from self._updates

    def get_state(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        # Return a minimal values-like snapshot
        return {"values": {"ok": True}}


def test_facade_wrap_invokes_and_records(tmp_path: Path) -> None:
    # Minimal updates: a single updates chunk to produce one SYS event beyond input
    updates: list[tuple[str, Any]] = [("updates", {"state": {"ok": True}})]
    graph = _FakeGraph(updates)
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    handle = wrap(
        graph,
        project="p",
        name="facade",
        store=store,
        stream_modes=("updates",),
        snapshot_every=0,
        stream_subgraphs=False,
    )
    result = handle.invoke({"x": 1}, config={})
    assert handle.last_run_id is not None
    run_id = handle.last_run_id
    # Events persisted: at least initial SYS
    events = store.list_events(run_id)
    assert events and events[0].action_type.value == "SYS"
    # Facade returns either values-like or None depending on fake graph
    _ = result
