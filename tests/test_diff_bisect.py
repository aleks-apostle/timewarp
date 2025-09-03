from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from timewarp.diff import bisect_divergence
from timewarp.events import ActionType, BlobKind, Event, Run
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
    labels: dict[str, str] | None = None,
) -> Event:
    b = store.put_blob(run.run_id, step, kind, json.dumps(payload).encode())
    ev = Event(
        run_id=run.run_id,
        step=step,
        action_type=atype,
        actor=actor,
        output_ref=b,
        hashes={"output": b.sha256_hex},
        labels=labels or {},
    )
    store.append_event(ev)
    return ev


def test_bisect_simple_single_step_mismatch(tmp_path: Path) -> None:
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

    b = bisect_divergence(store, run_a.run_id, run_b.run_id)
    assert b is not None
    assert b["cause"] == "output hash mismatch"
    assert b["start_a"] == 1 and b["end_a"] == 1
    assert b["start_b"] == 1 and b["end_b"] == 1


def test_bisect_reordered_anchor_aligned(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")

    run_a = Run(project="p", name="a")
    run_b = Run(project="p", name="b")
    store.create_run(run_a)
    store.create_run(run_b)

    # Two events with distinct anchors: actors A and B, same payloads
    def _w(run: Run, step: int, actor: str) -> Event:
        return _write_event(
            store,
            run,
            step,
            {"val": actor},
            actor=actor,
            labels={"namespace": actor},
        )

    # A then B in run A
    _w(run_a, 0, "A")
    _w(run_a, 1, "B")
    # B then A in run B (reordered)
    _w(run_b, 0, "B")
    _w(run_b, 1, "A")

    # With anchor realignment, no semantic divergence (hashes/anchors match)
    b = bisect_divergence(store, run_a.run_id, run_b.run_id)
    assert b is None


def test_bisect_anchor_mismatch_cause(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")

    run_a = Run(project="p", name="a")
    run_b = Run(project="p", name="b")
    store.create_run(run_a)
    store.create_run(run_b)

    # A's first anchor is "A"; B starts with an unrelated anchor "X"
    _write_event(store, run_a, 0, {"val": "A"}, actor="A", labels={"namespace": "A"})
    _write_event(store, run_a, 1, {"val": "B"}, actor="B", labels={"namespace": "B"})

    _write_event(store, run_b, 0, {"val": "X"}, actor="X", labels={"namespace": "X"})
    _write_event(store, run_b, 1, {"val": "A"}, actor="A", labels={"namespace": "A"})

    # Use a very small window to avoid realignment from B[0] -> A[0]
    b = bisect_divergence(store, run_a.run_id, run_b.run_id, window=1)
    assert b is not None
    assert b["cause"] == "anchor mismatch"
    # Expect mismatch at the first steps
    assert b["start_a"] == 0 and b["start_b"] == 0
