from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from timewarp.cli import main as cli_main
from timewarp.events import ActionType, BlobKind, Event
from timewarp.store import LocalStore


def test_fsck_gc_respects_grace(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)

    run_id = uuid4()
    # Create a referenced blob and event
    ref = store.put_blob(run_id, 1, BlobKind.OUTPUT, b"hello")
    store.append_event(
        Event(run_id=run_id, step=1, action_type=ActionType.SYS, actor="graph", output_ref=ref)
    )

    # Create two orphan files under the same run: one old, one new
    orphan_old = blobs / (Path(ref.path).parent) / "orphan.bin"
    orphan_old.parent.mkdir(parents=True, exist_ok=True)
    orphan_old.write_bytes(b"junk")
    # Set mtime to far in the past
    os.utime(orphan_old, (0, 0))

    orphan_new = blobs / (Path(ref.path).parent) / "orphan2.bin"
    orphan_new.write_bytes(b"junk")
    # Recent mtime: leave as is

    # Run fsck with gc-orphans and a small grace (e.g., 5s)
    rc = cli_main([str(db), str(blobs), "fsck", str(run_id), "--gc-orphans", "--grace", "5"])
    assert rc == 0
    # Old orphan should be deleted; new orphan should still exist
    assert not orphan_old.exists()
    assert orphan_new.exists()
