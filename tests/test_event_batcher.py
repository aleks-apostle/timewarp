from __future__ import annotations

from datetime import UTC, datetime, timedelta

from timewarp.events import ActionType, Event
from timewarp.langgraph.recorder import EventBatcher


class _CaptureStore:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Event] | Event]] = []

    # Batch API used by batcher
    def append_events(self, events: list[Event]) -> None:  # pragma: no cover - trivial
        self.calls.append(("append_events", events))

    # Per-event fallback used by batcher on retry
    def append_event(self, event: Event) -> None:
        self.calls.append(("append_event", event))


def _ev(step: int, ts: datetime) -> Event:
    return Event(
        run_id=__import__("uuid").uuid4(),
        step=step,
        action_type=ActionType.SYS,
        actor="graph",
        ts=ts,
    )


def test_event_batcher_marks_ts_regression_and_batches() -> None:
    store = _CaptureStore()
    b = EventBatcher(store=store, batch_size=3)

    t0 = datetime.now(UTC)
    e0 = _ev(0, t0)
    e1 = _ev(1, t0 + timedelta(milliseconds=1))
    # Regressed timestamp
    e2 = _ev(2, t0 - timedelta(milliseconds=5))

    # Append three events to trigger a batch write
    b.append(e0)
    b.append(e1)
    b.append(e2)

    # One batched call with three events
    assert store.calls and store.calls[0][0] == "append_events"
    appended = store.calls[0][1]
    assert isinstance(appended, list) and len(appended) == 3
    # The regressed event should be labeled by the batcher
    reg = appended[2]
    assert isinstance(reg, Event)
    assert reg.labels.get("ts_regressed") == "true"


def test_event_batcher_flush_writes_pending() -> None:
    store = _CaptureStore()
    b = EventBatcher(store=store, batch_size=5)
    t0 = datetime.now(UTC)
    for i in range(3):
        b.append(_ev(i, t0 + timedelta(milliseconds=i)))
    # No automatic batch write yet (batch_size not reached)
    assert not store.calls
    # Flush should write remaining
    b.flush()
    assert store.calls and store.calls[0][0] == "append_events"
    pend = store.calls[0][1]
    assert isinstance(pend, list) and len(pend) == 3
