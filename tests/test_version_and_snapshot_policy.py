from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson as _orjson
import pytest

from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.replay import Replay, SchemaMismatch
from timewarp.store import LocalStore


def test_replay_version_guard_schema_mismatch(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="gv")
    store.create_run(run)
    # Create two events with differing schema_version
    b0 = store.put_blob(run.run_id, 0, BlobKind.INPUT, _orjson.dumps({"a": 1}))
    e0 = Event(
        run_id=run.run_id,
        step=0,
        action_type=ActionType.SYS,
        actor="graph",
        input_ref=b0,
        hashes={"input": b0.sha256_hex},
        schema_version=1,
    )
    store.append_event(e0)
    b1 = store.put_blob(run.run_id, 1, BlobKind.OUTPUT, _orjson.dumps({"x": 1}))
    e1 = Event(
        run_id=run.run_id,
        step=1,
        action_type=ActionType.SYS,
        actor="graph",
        output_ref=b1,
        hashes={"output": b1.sha256_hex},
        schema_version=2,
    )
    store.append_event(e1)
    with pytest.raises(SchemaMismatch):
        _ = Replay(store=store, run_id=run.run_id)


def test_replay_version_guard_adapter_mismatch(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="gav")
    store.create_run(run)
    b0 = store.put_blob(run.run_id, 0, BlobKind.INPUT, _orjson.dumps({"a": 1}))
    e0 = Event(
        run_id=run.run_id,
        step=0,
        action_type=ActionType.SYS,
        actor="graph",
        input_ref=b0,
        hashes={"input": b0.sha256_hex},
        model_meta={"adapter_version": "A"},
    )
    store.append_event(e0)
    b1 = store.put_blob(run.run_id, 1, BlobKind.OUTPUT, _orjson.dumps({"x": 1}))
    e1 = Event(
        run_id=run.run_id,
        step=1,
        action_type=ActionType.SYS,
        actor="graph",
        output_ref=b1,
        hashes={"output": b1.sha256_hex},
        model_meta={"adapter_version": "B"},
    )
    store.append_event(e1)
    with pytest.raises(SchemaMismatch):
        _ = Replay(store=store, run_id=run.run_id)


class _FakeValuesGraph:
    def __init__(self, updates: list[tuple[str, Any]]) -> None:
        self._updates = updates

    def stream(self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any):
        yield from self._updates


def test_snapshot_policy_prefers_values_stream(tmp_path: Path) -> None:
    # Prepare a fake stream with two values updates
    updates: list[tuple[str, Any]] = [
        ("values", {"state": {"x": 1}}),
        ("values", {"state": {"x": 2}}),
    ]
    graph = _FakeValuesGraph(updates)
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="snap", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        snapshot_every=1,  # snapshot after each streamed update
        stream_modes=("values",),
        stream_subgraphs=False,
    )
    _ = rec.invoke({"x": 0}, config={})
    events = store.list_events(run.run_id)
    # Expect: initial SYS, then for each values update a SYS event and a SNAPSHOT event
    # We had 2 values updates -> at least 1 snapshot due to snapshot_every=1
    assert any(e.action_type is ActionType.SNAPSHOT for e in events)
    # Latest snapshot should reflect the latest values state
    latest_snap = next(e for e in reversed(events) if e.action_type is ActionType.SNAPSHOT)
    raw = store.get_blob(latest_snap.output_ref) if latest_snap.output_ref else None
    assert raw is not None
    obj = _orjson.loads(raw)
    assert obj.get("state", {}).get("x") == 2
