from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _FakeGraphValues:
    def __init__(self, updates: list[tuple[str, Any]]) -> None:
        self._updates = updates

    def stream(
        self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any
    ) -> Iterable[Any]:
        # Yield provided updates directly
        yield from self._updates

    def get_state(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        # Minimal values-like snapshot shape
        return {"values": {"ok": True}}


def test_decision_snapshot_emitted_when_enabled(tmp_path: Path) -> None:
    # Two values updates with distinct `next` to trigger DECISION twice
    updates: list[tuple[str, Any]] = [
        ("values", {"values": {"next": ["n1"], "state": {"x": 1}}}),
        ("values", {"values": {"next": ["n2"], "state": {"x": 2}}}),
    ]
    graph = _FakeGraphValues(updates)
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="snap", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        snapshot_every=0,  # disable interval snapshots
        snapshot_on={"decision"},
        stream_modes=("values",),
        stream_subgraphs=False,
    )
    _ = rec.invoke({"input": 1}, config={"configurable": {"thread_id": "t1"}})

    events = store.list_events(run.run_id)
    # For each DECISION, verify the following event is a SNAPSHOT
    for idx, ev in enumerate(events):
        if ev.action_type is ActionType.DECISION:
            assert idx + 1 < len(events), "DECISION must be followed by a SNAPSHOT"
            nxt = events[idx + 1]
            assert nxt.action_type is ActionType.SNAPSHOT


def test_state_pruner_applied_on_snapshot(tmp_path: Path) -> None:
    big = "A" * 2000
    # Two updates: put the large payload in the FIRST so that the snapshot on the
    # SECOND decision uses last_values with the large payload
    updates: list[tuple[str, Any]] = [
        ("values", {"values": {"next": ["n1"], "state": {"big": big}}}),
        ("values", {"values": {"next": ["n2"], "state": {"big": "B" * 10}}}),
    ]
    graph = _FakeGraphValues(updates)
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="prune", framework="langgraph")

    def pruner(state: Any) -> Any:
        # Truncate any string values to <= 50 chars for the test
        def walk(x: Any) -> Any:
            if isinstance(x, str):
                return x[:50]
            if isinstance(x, dict):
                return {k: walk(v) for k, v in x.items()}
            if isinstance(x, list):
                return [walk(v) for v in x]
            return x

        return walk(state)

    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        snapshot_every=0,
        snapshot_on={"decision"},
        state_pruner=pruner,
        stream_modes=("values",),
        stream_subgraphs=False,
    )
    _ = rec.invoke({"input": 1}, config={"configurable": {"thread_id": "t1"}})

    events = store.list_events(run.run_id)
    # Locate a snapshot event and verify payload was pruned
    # Choose the snapshot that immediately follows a DECISION
    # Choose the snapshot following the LAST decision (ensures last_values populated)
    snap = None
    for idx in range(len(events) - 2, -1, -1):
        ev = events[idx]
        if ev.action_type is ActionType.DECISION and idx + 1 < len(events):
            cand = events[idx + 1]
            if cand.action_type is ActionType.SNAPSHOT:
                snap = cand
                break
    assert snap is not None and snap.output_ref is not None
    raw = store.get_blob(snap.output_ref)
    from timewarp.codec import from_bytes as _from_bytes

    obj = _from_bytes(raw)
    # State-like payload should exist and contain truncated string
    assert isinstance(obj, dict) and "values" in obj
    # Verify the big string under values.state.big was truncated by the pruner
    state = obj.get("values", {}).get("state", {})  # type: ignore[assignment]
    assert isinstance(state, dict)
    big_val = state.get("big")
    assert isinstance(big_val, str) and len(big_val) <= 50
