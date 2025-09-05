from __future__ import annotations

from timewarp.cli.helpers.events import filter_events
from timewarp.cli.helpers.filters import parse_list_filters
from timewarp.events import ActionType, Event


def _evt(step: int, kind: ActionType, actor: str, labels: dict[str, str]) -> Event:
    from uuid import uuid4

    return Event(run_id=uuid4(), step=step, action_type=kind, actor=actor, labels=labels)


def test_parse_and_filter_events() -> None:
    events: list[Event] = [
        _evt(0, ActionType.SYS, "graph", {"thread_id": "t1"}),
        _evt(1, ActionType.LLM, "compose", {"thread_id": "t1", "namespace": "a"}),
        _evt(2, ActionType.TOOL, "tooler", {"thread_id": "t2", "namespace": "b"}),
    ]
    filters = parse_list_filters(["type=LLM", "node=compose", "thread=t1"])  # type: ignore[arg-type]
    out = filter_events(
        events,
        etype=filters.get("type"),
        node=filters.get("node"),
        thread=filters.get("thread"),
        namespace=filters.get("namespace"),
    )
    assert len(out) == 1 and out[0].action_type is ActionType.LLM and out[0].actor == "compose"
