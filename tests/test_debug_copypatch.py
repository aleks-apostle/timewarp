from __future__ import annotations

from pathlib import Path

from timewarp.codec import to_bytes
from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.store import LocalStore


def test_copypatch_like_dump(tmp_path: Path) -> None:
    db = tmp_path / "tw.sqlite3"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)
    run = Run(project="t", name="n", framework="langgraph")
    store.create_run(run)
    step = 0
    payload = {"hello": "world", "n": 1}
    out_ref = store.put_blob(run.run_id, step, BlobKind.OUTPUT, to_bytes(payload))
    ev = Event(
        run_id=run.run_id,
        step=step,
        action_type=ActionType.SYS,
        actor="test",
        output_ref=out_ref,
        hashes={"output": out_ref.sha256_hex},
    )
    store.append_event(ev)

    # Use the CLI helper to dump to file (used by copypatch command)
    from timewarp.cli.helpers.state import dump_event_output_to_file

    out_file = tmp_path / "patches" / f"alt_{step}.json"
    dump_event_output_to_file(store, ev, out_file)
    assert out_file.exists()
    txt = out_file.read_text()
    assert "hello" in txt and "world" in txt
