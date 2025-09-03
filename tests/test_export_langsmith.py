from __future__ import annotations

from pathlib import Path
from typing import Any

from timewarp.codec import to_bytes
from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.exporters.langsmith import export_run, serialize_run
from timewarp.store import LocalStore


def test_serialize_run_with_inline_blobs(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="n", framework="langgraph")
    store.create_run(run)

    # Create a minimal event with input/output blobs
    step = 0
    inp_ref = store.put_blob(run.run_id, step, BlobKind.INPUT, to_bytes({"hello": "world"}))
    out_ref = store.put_blob(run.run_id, step, BlobKind.OUTPUT, to_bytes({"ok": True}))
    ev = Event(
        run_id=run.run_id,
        step=step,
        action_type=ActionType.SYS,
        actor="graph",
        input_ref=inp_ref,
        output_ref=out_ref,
        labels={"node": "graph"},
        hashes={},
    )
    store.append_event(ev)

    payload = serialize_run(store, run.run_id, include_blobs=True)
    assert set(payload.keys()) == {"run", "events"}
    assert payload["run"]["run_id"]
    assert isinstance(payload["events"], list) and len(payload["events"]) == 1
    e0 = payload["events"][0]
    # Presence of inline payloads when include_blobs=True
    assert e0.get("input_payload") == {"hello": "world"}
    assert e0.get("output_payload") == {"ok": True}


def test_export_run_with_stub_client(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="export", framework="langgraph")
    store.create_run(run)
    # Append a small SYS event without blobs as a smoke test
    ev = Event(
        run_id=run.run_id,
        step=0,
        action_type=ActionType.SYS,
        actor="graph",
        labels={},
        hashes={},
    )
    store.append_event(ev)

    class _Client:
        def __init__(self) -> None:
            self._last: dict[str, Any] | None = None

        def create_run(self, payload: dict[str, Any]) -> None:
            self._last = payload

    client = _Client()
    payload = export_run(store, run.run_id, client=client)
    assert isinstance(payload, dict) and payload.get("run")
    assert client._last == payload
