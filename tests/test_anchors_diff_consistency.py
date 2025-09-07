from __future__ import annotations

from uuid import uuid4

from timewarp.diff import make_anchor_key
from timewarp.events import ActionType, Event
from timewarp.langgraph.anchors import make_anchor_id


def test_adapter_anchor_matches_diff_anchor_key() -> None:
    labels: dict[str, str] = {
        "thread_id": "t-123",
        "namespace": "agent/main",
        "node": "agent",
    }
    actor = "agent"
    aid = make_anchor_id(ActionType.LLM, actor, dict(labels))
    # Attach anchor id label as the adapter does
    labels["anchor_id"] = aid

    ev1 = Event(
        run_id=uuid4(),
        step=1,
        action_type=ActionType.LLM,
        actor=actor,
        labels=dict(labels),
    )
    ev2 = Event(
        run_id=uuid4(),
        step=2,
        action_type=ActionType.LLM,
        actor=actor,
        labels=dict(labels),
    )

    # Diff should prefer explicit anchor id and produce a comparable key
    k1 = make_anchor_key(ev1)
    k2 = make_anchor_key(ev2)
    assert k1 == ("ANCHOR", aid)
    assert k1 == k2


def test_tool_anchor_includes_tool_name() -> None:
    labels: dict[str, str] = {
        "thread_id": "t-xyz",
        "namespace": "tools/search",
        "node": "tools",
    }
    actor = "tools"
    aid = make_anchor_id(ActionType.TOOL, actor, dict(labels), tool_name="search")
    labels["anchor_id"] = aid

    ev = Event(
        run_id=uuid4(),
        step=10,
        action_type=ActionType.TOOL,
        actor=actor,
        labels=dict(labels),
    )
    assert make_anchor_key(ev) == ("ANCHOR", aid)
