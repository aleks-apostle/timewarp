from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from timewarp.determinism import freeze_time_at
from timewarp.events import ActionType, Event


def test_event_ts_uses_frozen_time() -> None:
    t = datetime(2020, 1, 2, 3, 4, 5, tzinfo=UTC)
    with freeze_time_at(t):
        ev = Event(
            run_id=uuid4(),
            step=1,
            action_type=ActionType.SYS,
            actor="graph",
        )
        assert ev.ts == t
