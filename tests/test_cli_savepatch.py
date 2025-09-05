from __future__ import annotations

from pathlib import Path

import orjson as _orjson

from timewarp.cli.helpers.state import dump_event_output_to_file
from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.store import LocalStore


def test_dump_event_output_to_file(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)
    run = Run(project="p", name="patch")
    store.create_run(run)
    payload = {"ok": True, "msg": "hello"}
    bref = store.put_blob(run.run_id, 1, BlobKind.OUTPUT, _orjson.dumps(payload))
    ev = Event(
        run_id=run.run_id, step=1, action_type=ActionType.SYS, actor="graph", output_ref=bref
    )
    out_path = tmp_path / "out.json"
    dump_event_output_to_file(store, ev, out_path)
    data = _orjson.loads(out_path.read_bytes())
    assert data == payload
