from __future__ import annotations

from uuid import uuid4

from timewarp.diff import make_anchor_key, realign_by_anchor
from timewarp.events import ActionType, Event


def _e(step: int, kind: ActionType, actor: str, labels: dict[str, str]) -> Event:
    return Event(run_id=uuid4(), step=step, action_type=kind, actor=actor, labels=labels)


def test_make_anchor_key_uses_anchor_id_when_present() -> None:
    e = _e(0, ActionType.SYS, "graph", {"anchor_id": "t:node:ns:SYS"})
    k = make_anchor_key(e)
    assert isinstance(k, tuple) and k[0] == "ANCHOR"


def test_realign_by_anchor_finds_within_window() -> None:
    a: list[Event] = [
        _e(0, ActionType.SYS, "A", {"namespace": "A"}),
        _e(1, ActionType.SYS, "B", {"namespace": "B"}),
        _e(2, ActionType.SYS, "C", {"namespace": "C"}),
    ]
    b: list[Event] = [
        _e(0, ActionType.SYS, "B", {"namespace": "B"}),
        _e(1, ActionType.SYS, "A", {"namespace": "A"}),
        _e(2, ActionType.SYS, "C", {"namespace": "C"}),
    ]
    mi, mj = realign_by_anchor(a, b, start_a=0, start_b=0, window=2)
    assert mi is not None or mj is not None
