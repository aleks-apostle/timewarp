from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

from ..events import Event
from ..store import LocalStore


@dataclass
class EventBatcher:
    store: LocalStore
    batch_size: int = 1

    _pending_events: list[Event] = field(default_factory=list, init=False, repr=False)
    _pending_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _last_ts: datetime | None = field(default=None, init=False, repr=False)

    def append(self, ev: Event) -> None:
        # Monotonic timestamp validation; mark skew as a label when detected
        try:
            if self._last_ts is not None and ev.ts < self._last_ts:
                new_labels = dict(ev.labels or {})
                new_labels["ts_regressed"] = "true"
                ev = ev.model_copy(update={"labels": new_labels})
            if self._last_ts is None or ev.ts > self._last_ts:
                self._last_ts = ev.ts
        except Exception:
            pass

        bs = max(1, int(self.batch_size))
        if bs == 1:
            self._append_events_with_retry([ev])
            return
        with self._pending_lock:
            self._pending_events.append(ev)
            if len(self._pending_events) < bs:
                return
            batch = list(self._pending_events)
            self._pending_events.clear()
        try:
            self._append_events_with_retry(batch)
        except Exception:
            with self._pending_lock:
                self._pending_events[:0] = batch
            raise

    def flush(self) -> None:
        with self._pending_lock:
            if not self._pending_events:
                return
            batch = list(self._pending_events)
            self._pending_events.clear()
        try:
            self._append_events_with_retry(batch)
        except Exception:
            with self._pending_lock:
                self._pending_events[:0] = batch
            raise

    def _append_events_with_retry(self, events: list[Event]) -> None:
        if not events:
            return
        import time as _time

        backoffs = (0.05, 0.1, 0.2)
        for i, delay in enumerate(backoffs):
            try:
                self.store.append_events(events)
                return
            except Exception:
                if i == len(backoffs) - 1:
                    break
                try:
                    _time.sleep(delay)
                except Exception:
                    pass
        for ev in events:
            appended = False
            for j, delay2 in enumerate(backoffs):
                try:
                    self.store.append_event(ev)
                    appended = True
                    break
                except Exception:
                    if j == len(backoffs) - 1:
                        break
                    try:
                        _time.sleep(delay2)
                    except Exception:
                        pass
            if not appended:
                raise
