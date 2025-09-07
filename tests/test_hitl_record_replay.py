from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from timewarp.codec import from_bytes
from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.replay import LangGraphReplayer
from timewarp.store import LocalStore


class _FakeHITLGraph:
    """Minimal fake graph exposing a LangGraph-like .stream and .get_state."""

    def __init__(self, updates: list[Any] | None = None) -> None:
        self._updates = updates or []

    def stream(self, inputs: dict[str, Any], config: dict[str, Any], **_: Any) -> Iterable[Any]:
        # Emit a small sequence of updates including an interrupt envelope
        yield ("values", {"values": {"foo": 1}})
        yield ("updates", {"__interrupt__": {"reason": "human"}})
        yield ("values", {"values": {"foo": 2}})

    def get_state(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"values": {"foo": 2}}


def test_hitl_event_record_and_envelope(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "tw.sqlite3", blobs_root=tmp_path / "blobs")
    graph = _FakeHITLGraph()
    run = Run(project="test", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        stream_modes=("updates", "values"),
        stream_subgraphs=False,
        snapshot_on={"terminal"},
    )
    _ = rec.invoke({"x": 1}, config={})

    events = store.list_events(run.run_id)
    kinds = [e.action_type for e in events]
    assert ActionType.HITL in kinds, "HITL event not recorded"
    hitl = next(e for e in events if e.action_type is ActionType.HITL)
    assert hitl.output_ref is not None
    payload = from_bytes(store.get_blob(hitl.output_ref))
    assert isinstance(payload, dict)
    assert payload.get("hitl", {}).get("type") == "interrupt"
    assert "payload" in payload.get("hitl", {})


def test_hitl_does_not_block_replay_resume(tmp_path: Path) -> None:
    # Record with HITL
    store = LocalStore(db_path=tmp_path / "tw.sqlite3", blobs_root=tmp_path / "blobs")
    run = Run(project="test", framework="langgraph")
    rec = LangGraphRecorder(
        graph=_FakeHITLGraph(),
        store=store,
        run=run,
        stream_modes=("updates", "values"),
        stream_subgraphs=False,
        snapshot_on={"terminal"},
    )
    _ = rec.invoke({"x": 1}, config={})

    # Resume with a no-op stub graph; should not require wrappers since no LLM/TOOL.
    class _NoopGraph:
        def stream(self, inputs: dict[str, Any], config: dict[str, Any], **_: Any) -> Iterable[Any]:
            yield ("values", {"values": {"foo": 2}})

        def get_state(self, config: dict[str, Any]) -> dict[str, Any]:
            return {"values": {"foo": 2}}

    replayer = LangGraphReplayer(graph=_NoopGraph(), store=store)
    session = replayer.resume(run.run_id, from_step=None, thread_id=None, install_wrappers=None)
    assert session.result == {"foo": 2}
