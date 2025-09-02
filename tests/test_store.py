from __future__ import annotations

import json
from pathlib import Path

from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.store import LocalStore


def test_store_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "tw.db"
    blobs = tmp_path / "blobs"
    st = LocalStore(db_path=db, blobs_root=blobs)

    run = Run(project="p", name="n")
    st.create_run(run)

    payload = json.dumps({"hello": "world"}).encode()
    bref = st.put_blob(run.run_id, 0, BlobKind.INPUT, payload)

    ev = Event(
        run_id=run.run_id,
        step=0,
        action_type=ActionType.SYS,
        actor="graph",
        input_ref=bref,
        hashes={"input": bref.sha256_hex},
    )
    st.append_event(ev)

    events = st.list_events(run.run_id)
    assert len(events) == 1
    got = events[0]
    assert got.run_id == ev.run_id
    assert got.step == 0
    # Load blob
    raw = st.get_blob(bref)
    assert raw == payload
