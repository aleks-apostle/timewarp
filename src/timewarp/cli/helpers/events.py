from __future__ import annotations

from collections.abc import Iterable

from ...events import Event


def filter_events(
    events: Iterable[Event],
    *,
    etype: str | None,
    node: str | None,
    thread: str | None,
    namespace: str | None,
    tool_kind: str | None = None,
    tool_name: str | None = None,
) -> list[Event]:
    def _ok(e: Event) -> bool:
        if etype and e.action_type.value != etype:
            return False
        if node and e.actor != node and e.labels.get("node") != node:
            return False
        if thread and e.labels.get("thread_id") != thread:
            return False
        if namespace and e.labels.get("namespace") != namespace:
            return False
        if tool_kind and (e.tool_kind or "") != tool_kind:
            return False
        if tool_name and (e.tool_name or "") != tool_name:
            return False
        return True

    return [e for e in events if _ok(e)]
