from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from timewarp.events import ActionType, Event
from timewarp.store import LocalStore


def _mk_event(step: int) -> Event:
    return Event(
        run_id=uuid4(),
        step=step,
        action_type=ActionType.SYS,
        actor="graph",
        ts=datetime.now(UTC),
    )


def test_append_event_monotonic_guard(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)

    # Create run with two events
    from timewarp.events import Event as E

    run_id = uuid4()
    e1 = E(run_id=run_id, step=1, action_type=ActionType.SYS, actor="graph")
    e2 = E(run_id=run_id, step=2, action_type=ActionType.SYS, actor="graph")
    store.append_event(e1)
    store.append_event(e2)

    # Attempt to append with lower step should raise
    e0 = E(run_id=run_id, step=2, action_type=ActionType.SYS, actor="graph")
    with pytest.raises(RuntimeError):
        store.append_event(e0)


def test_append_events_batch_monotonic_guard(tmp_path: Path) -> None:
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)

    run_id = uuid4()
    # First batch OK
    store.append_events(
        [
            Event(run_id=run_id, step=1, action_type=ActionType.SYS, actor="graph"),
            Event(run_id=run_id, step=2, action_type=ActionType.SYS, actor="graph"),
        ]
    )
    # Second batch with non-increasing order should fail
    with pytest.raises(RuntimeError):
        store.append_events(
            [
                Event(run_id=run_id, step=4, action_type=ActionType.SYS, actor="graph"),
                Event(run_id=run_id, step=3, action_type=ActionType.SYS, actor="graph"),
            ]
        )
    # Second batch with first step <= existing max should fail
    with pytest.raises(RuntimeError):
        store.append_events(
            [Event(run_id=run_id, step=2, action_type=ActionType.SYS, actor="graph")]
        )
