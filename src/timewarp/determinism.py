from __future__ import annotations

import pickle
import random
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Protocol


class TimeProvider(Protocol):
    def now(self) -> datetime: ...


class SystemTimeProvider:
    """Default time provider using system UTC time."""

    def now(self) -> datetime:
        return datetime.now(UTC)


# Process-global provider (overridable by apps/tests)
_provider: TimeProvider | None = None


def set_time_provider(provider: TimeProvider | None) -> None:
    global _provider
    _provider = provider


def get_time_provider() -> TimeProvider:
    return _provider or SystemTimeProvider()


# Contextual frozen times (thread/context-local stack)
# Use immutable tuple for ContextVar default to satisfy linting and avoid shared mutation.
_frozen_stack: ContextVar[tuple[datetime, ...]] = ContextVar("tw_frozen_times", default=())


def now() -> datetime:
    stack = _frozen_stack.get()
    if stack:
        return stack[-1]
    return get_time_provider().now()


@contextmanager
def freeze_time_at(ts: datetime) -> Iterator[None]:
    """Temporarily freeze `now()` to the provided timestamp within the context."""
    stack = _frozen_stack.get()
    new_stack = (*stack, ts)
    token = _frozen_stack.set(new_stack)
    try:
        yield
    finally:
        try:
            curr = _frozen_stack.get()
            if curr:
                _frozen_stack.set(curr[:-1])
            else:
                _frozen_stack.reset(token)
        except Exception:
            _frozen_stack.reset(token)


def snapshot_rng() -> bytes:
    """Capture Python RNG state as bytes.

    Uses pickle for fidelity; caller stores this in the event.
    """

    state = random.getstate()
    return pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)


def restore_rng(state_bytes: bytes) -> None:
    """Restore Python RNG from previously captured bytes."""

    state = pickle.loads(state_bytes)
    random.setstate(state)
