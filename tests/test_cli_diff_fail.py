from __future__ import annotations

from pathlib import Path
from uuid import UUID

import orjson as _orjson

from timewarp.cli import main as cli_main
from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.store import LocalStore


def _mk_run(
    store: LocalStore, name: str, payload1: dict[str, object], payload2: dict[str, object]
) -> UUID:
    run = Run(project="p", name=name)
    store.create_run(run)
    b0 = store.put_blob(run.run_id, 0, BlobKind.INPUT, _orjson.dumps({"a": 1}))
    e0 = Event(
        run_id=run.run_id,
        step=0,
        action_type=ActionType.SYS,
        actor="graph",
        input_ref=b0,
        hashes={"input": b0.sha256_hex},
    )
    store.append_event(e0)
    b1 = store.put_blob(run.run_id, 1, BlobKind.OUTPUT, _orjson.dumps(payload1))
    e1 = Event(
        run_id=run.run_id,
        step=1,
        action_type=ActionType.SYS,
        actor="graph",
        output_ref=b1,
        hashes={"output": b1.sha256_hex},
    )
    store.append_event(e1)
    b2 = store.put_blob(run.run_id, 2, BlobKind.OUTPUT, _orjson.dumps(payload2))
    e2 = Event(
        run_id=run.run_id,
        step=2,
        action_type=ActionType.SNAPSHOT,
        actor="graph",
        output_ref=b2,
        hashes={"state": b2.sha256_hex},
    )
    store.append_event(e2)
    return run.run_id


def test_cli_diff_fail_on_divergence(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)
    run_a = _mk_run(store, "A", {"x": 1}, {"s": 1})
    run_b = _mk_run(store, "B", {"x": 2}, {"s": 1})

    rc = cli_main([str(db), str(blobs), "diff", str(run_a), str(run_b), "--fail-on-divergence"])
    assert rc == 1
