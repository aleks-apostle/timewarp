from __future__ import annotations

from pathlib import Path
from uuid import UUID

import orjson as _orjson

from timewarp.cli import main as cli_main
from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.store import LocalStore


def _mk_run(store: LocalStore, name: str) -> UUID:
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
    b1 = store.put_blob(run.run_id, 1, BlobKind.OUTPUT, _orjson.dumps({"x": 1}))
    e1 = Event(
        run_id=run.run_id,
        step=1,
        action_type=ActionType.SYS,
        actor="graph",
        output_ref=b1,
        hashes={"output": b1.sha256_hex},
    )
    store.append_event(e1)
    return run.run_id


def test_cli_list_json_outputs_expected_fields(tmp_path: Path, capsys) -> None:
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)
    run_a = _mk_run(store, "A")
    # Create a second run and tag branch_of for lineage
    run_b = Run(project="p", name="B", labels={"branch_of": str(run_a)})
    store.create_run(run_b)
    b0 = store.put_blob(run_b.run_id, 0, BlobKind.INPUT, _orjson.dumps({"a": 2}))
    e0 = Event(
        run_id=run_b.run_id,
        step=0,
        action_type=ActionType.SYS,
        actor="graph",
        input_ref=b0,
        hashes={"input": b0.sha256_hex},
        labels={"branch_of": str(run_a)},
    )
    store.append_event(e0)

    rc = cli_main([str(db), str(blobs), "list", "--json"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    rows = _orjson.loads(out)
    assert isinstance(rows, list)
    # Two rows at least, and entries have expected keys
    assert any(r["run_id"] == str(run_a) for r in rows)
    # Verify fields presence and types for the run with branch_of
    entry_b = next(r for r in rows if r["run_id"] == str(run_b.run_id))
    assert entry_b["project"] == "p"
    assert entry_b["name"] == "B"
    assert isinstance(entry_b["started_at"], str)
    assert isinstance(entry_b["events"], int)
    # branch_of should be present for the forked run
    assert entry_b["branch_of"] == str(run_a)
