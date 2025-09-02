from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from timewarp.diff import first_divergence
from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.replay import Replay
from timewarp.store import LocalStore


def _write_event(
    store: LocalStore,
    run: Run,
    step: int,
    payload: dict[str, Any],
    *,
    kind: BlobKind = BlobKind.OUTPUT,
    atype: ActionType = ActionType.SYS,
    actor: str = "graph",
) -> Event:
    b = store.put_blob(run.run_id, step, kind, json.dumps(payload).encode())
    ev = Event(
        run_id=run.run_id,
        step=step,
        action_type=atype,
        actor=actor,
        output_ref=b,
        hashes={"output": b.sha256_hex},
    )
    store.append_event(ev)
    return ev


def test_first_divergence_and_replay(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")

    run_a = Run(project="p", name="a")
    run_b = Run(project="p", name="b")
    store.create_run(run_a)
    store.create_run(run_b)

    _write_event(store, run_a, 0, {"val": 1})
    _write_event(store, run_b, 0, {"val": 1})

    # Diverge at step 1
    _write_event(store, run_a, 1, {"val": 2})
    _write_event(store, run_b, 1, {"val": 3})

    d = first_divergence(store, run_a.run_id, run_b.run_id)
    assert d is not None
    assert d.step_a == 1 and d.step_b == 1
    assert "output hash mismatch" in d.reason

    # Replay: add a snapshot to inspect
    snap = store.put_blob(run_a.run_id, 2, BlobKind.STATE, json.dumps({"state": "ok"}).encode())
    store.append_event(
        Event(
            run_id=run_a.run_id,
            step=2,
            action_type=ActionType.SNAPSHOT,
            actor="graph",
            output_ref=snap,
            hashes={"state": snap.sha256_hex},
        )
    )

    rep = Replay(store=store, run_id=run_a.run_id)
    rep.goto(3)  # move past snapshot
    state = rep.inspect_state()
    assert isinstance(state, dict) and state.get("state") == "ok"


def test_first_divergence_anchor_reordering(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")

    run_a = Run(project="p", name="a")
    run_b = Run(project="p", name="b")
    store.create_run(run_a)
    store.create_run(run_b)

    # Two events with distinct anchors: actors A and B, same payloads
    def _w(run: Run, step: int, actor: str) -> Event:
        b = store.put_blob(run.run_id, step, BlobKind.OUTPUT, json.dumps({"val": actor}).encode())
        ev = Event(
            run_id=run.run_id,
            step=step,
            action_type=ActionType.SYS,
            actor=actor,
            output_ref=b,
            hashes={"output": b.sha256_hex},
            labels={"namespace": actor},
        )
        store.append_event(ev)
        return ev

    # A then B in run A
    _w(run_a, 0, "A")
    _w(run_a, 1, "B")
    # B then A in run B (reordered)
    _w(run_b, 0, "B")
    _w(run_b, 1, "A")

    # With anchor realignment, no semantic divergence (hashes/anchors match)
    d = first_divergence(store, run_a.run_id, run_b.run_id)
    assert d is None


def test_first_divergence_adapter_schema_mismatch(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run_a = Run(project="p", name="a")
    run_b = Run(project="p", name="b")
    store.create_run(run_a)
    store.create_run(run_b)

    b1 = store.put_blob(run_a.run_id, 0, BlobKind.OUTPUT, json.dumps({"x": 1}).encode())
    store.append_event(
        Event(
            run_id=run_a.run_id,
            step=0,
            action_type=ActionType.SYS,
            actor="graph",
            output_ref=b1,
            hashes={"output": b1.sha256_hex},
            schema_version=1,
            model_meta={"adapter_version": "1.0"},
        )
    )
    b2 = store.put_blob(run_b.run_id, 0, BlobKind.OUTPUT, json.dumps({"x": 1}).encode())
    store.append_event(
        Event(
            run_id=run_b.run_id,
            step=0,
            action_type=ActionType.SYS,
            actor="graph",
            output_ref=b2,
            hashes={"output": b2.sha256_hex},
            schema_version=1,
            model_meta={"adapter_version": "2.0"},
        )
    )
    d = first_divergence(store, run_a.run_id, run_b.run_id)
    assert d is not None and "adapter_version mismatch" in d.reason
