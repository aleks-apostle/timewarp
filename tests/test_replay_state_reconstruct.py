from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson as _orjson

from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.replay import Replay
from timewarp.store import LocalStore


def _blob(store: LocalStore, run_id, step: int, kind: BlobKind, obj: Any) -> tuple[bytes, str]:
    b = store.put_blob(run_id, step, kind, _orjson.dumps(obj))
    return _orjson.dumps(obj), b.sha256_hex


def test_state_reconstruction_from_snapshot_and_updates(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="recon")
    store.create_run(run)

    # Initial SYS input (not a state snapshot)
    b0, h0 = _blob(store, run.run_id, 0, BlobKind.INPUT, {"input": 1})
    store.append_event(
        Event(
            run_id=run.run_id,
            step=0,
            action_type=ActionType.SYS,
            actor="graph",
            input_ref=store.put_blob(run.run_id, 0, BlobKind.INPUT, b0),
            hashes={"input": h0},
        )
    )

    # Snapshot at step 1
    base = {"x": 1, "nested": {"a": 1}}
    b1, h1 = _blob(store, run.run_id, 1, BlobKind.STATE, base)
    store.append_event(
        Event(
            run_id=run.run_id,
            step=1,
            action_type=ActionType.SNAPSHOT,
            actor="graph",
            output_ref=store.put_blob(run.run_id, 1, BlobKind.STATE, b1),
            hashes={"state": h1},
        )
    )

    # Update values at step 2: {"values": {"x": 2}}
    u1_obj = {"values": {"x": 2}}
    b2, h2 = _blob(store, run.run_id, 2, BlobKind.OUTPUT, u1_obj)
    store.append_event(
        Event(
            run_id=run.run_id,
            step=2,
            action_type=ActionType.SYS,
            actor="graph",
            output_ref=store.put_blob(run.run_id, 2, BlobKind.OUTPUT, b2),
            hashes={"output": h2},
            labels={"stream_mode": "updates"},
        )
    )

    # Update values at step 3: { node: {"nested": {"b": 2}} }
    u2_obj = {"node": {"nested": {"b": 2}}}
    b3, h3 = _blob(store, run.run_id, 3, BlobKind.OUTPUT, u2_obj)
    store.append_event(
        Event(
            run_id=run.run_id,
            step=3,
            action_type=ActionType.SYS,
            actor="graph",
            output_ref=store.put_blob(run.run_id, 3, BlobKind.OUTPUT, b3),
            hashes={"output": h3},
            labels={"stream_mode": "updates"},
        )
    )

    rep = Replay(store=store, run_id=run.run_id)
    rep.goto(4)
    state = rep.inspect_state()
    assert isinstance(state, dict)
    assert state["x"] == 2
    assert state["nested"]["a"] == 1 and state["nested"]["b"] == 2


def test_state_reconstruction_respects_skip_and_overlay(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="recon2")
    store.create_run(run)

    # Snapshot at step 0
    base = {"x": 1}
    b0, h0 = _blob(store, run.run_id, 0, BlobKind.STATE, base)
    store.append_event(
        Event(
            run_id=run.run_id,
            step=0,
            action_type=ActionType.SNAPSHOT,
            actor="graph",
            output_ref=store.put_blob(run.run_id, 0, BlobKind.STATE, b0),
            hashes={"state": h0},
        )
    )

    # Update at step 1 (to be skipped)
    u1 = {"values": {"x": 2}}
    b1, h1 = _blob(store, run.run_id, 1, BlobKind.OUTPUT, u1)
    store.append_event(
        Event(
            run_id=run.run_id,
            step=1,
            action_type=ActionType.SYS,
            actor="graph",
            output_ref=store.put_blob(run.run_id, 1, BlobKind.OUTPUT, b1),
            hashes={"output": h1},
            labels={"stream_mode": "updates"},
        )
    )

    # Update at step 2 (overlay will replace)
    u2 = {"values": {"x": 3}}
    b2, h2 = _blob(store, run.run_id, 2, BlobKind.OUTPUT, u2)
    store.append_event(
        Event(
            run_id=run.run_id,
            step=2,
            action_type=ActionType.SYS,
            actor="graph",
            output_ref=store.put_blob(run.run_id, 2, BlobKind.OUTPUT, b2),
            hashes={"output": h2},
            labels={"stream_mode": "updates"},
        )
    )

    rep = Replay(store=store, run_id=run.run_id)
    rep.goto(3)
    rep.skip(1)  # ignore x=2 update
    rep.inject(2, {"values": {"x": 10}})  # replace x with 10
    state = rep.inspect_state()
    assert isinstance(state, dict)
    assert state["x"] == 10
