from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _ValuesGraph:
    def __init__(self, updates: list[tuple[str, Any]]) -> None:
        self._updates = updates

    def stream(
        self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any
    ) -> Iterable[Any]:
        yield from self._updates


def test_memory_events_emitted_for_configured_keys(tmp_path: Path) -> None:
    # Two values updates: first with mem.long payload, second unchanged, third changed
    updates: list[tuple[str, Any]] = [
        ("values", {"values": {"mem": {"long": {"a": 1, "b": 2}}}}),
        ("values", {"values": {"mem": {"long": {"a": 1, "b": 2}}}}),
        ("values", {"values": {"mem": {"long": {"a": 2, "b": 3}}}}),
    ]
    graph = _ValuesGraph(updates)
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="mem", framework="langgraph")

    # Memory pruner that keeps only key 'a'
    def pruner(x: Any) -> Any:
        if isinstance(x, dict):
            return {"a": x.get("a")}
        return x

    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        stream_modes=("values",),
        stream_subgraphs=False,
        memory_paths=("mem.long",),
        memory_pruner=pruner,
    )
    _ = rec.invoke({"x": 1}, config={"configurable": {"thread_id": "t-1"}})

    events = store.list_events(run.run_id)
    mems = [e for e in events if e.action_type is ActionType.MEMORY]
    # Expect two memory events: first PUT, then UPDATE (second identical update suppressed)
    assert len(mems) == 2
    assert mems[0].labels.get("mem_op") == "PUT"
    assert mems[1].labels.get("mem_op") == "UPDATE"
    # Scope inferred from path name
    assert mems[0].labels.get("mem_scope") == "long"
    # Thread label and node default present
    assert mems[0].labels.get("thread_id") == "t-1"
    # Provider metadata attached
    assert mems[0].mem_provider == "LangGraphState"
