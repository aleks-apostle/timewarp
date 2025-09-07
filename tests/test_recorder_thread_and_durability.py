from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from timewarp.events import Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _DurabilityGraph:
    """Graph capturing stream kwargs for assertions."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def stream(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[Any]:
        self.last_kwargs = dict(kwargs)
        # Emit one trivial updates chunk to advance recorder
        yield ("updates", {"ok": True})


def test_require_thread_id_guard_raises(tmp_path: Path) -> None:
    graph = _DurabilityGraph()
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="guard", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        require_thread_id=True,
    )
    with pytest.raises(ValueError):
        _ = rec.invoke({"x": 1}, config={})


def test_durability_default_sync_with_thread_id(tmp_path: Path) -> None:
    graph = _DurabilityGraph()
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="dur", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        # durability=None + thread_id → default to "sync"
        durability=None,
    )
    cfg = {"configurable": {"thread_id": "t-1"}}
    _ = rec.invoke({"x": 1}, config=cfg)
    # Initial event labeled with durability, and stream kwargs include durability
    events = store.list_events(run.run_id)
    assert events and events[0].labels.get("durability") == "sync"
    assert isinstance(graph.last_kwargs, dict)
    assert graph.last_kwargs.get("durability") == "sync"


def test_durability_override_passed_through(tmp_path: Path) -> None:
    graph = _DurabilityGraph()
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="dur2", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        durability="async",
    )
    cfg = {"configurable": {"thread_id": "t-1"}}
    _ = rec.invoke({"x": 1}, config=cfg)
    events = store.list_events(run.run_id)
    assert events and events[0].labels.get("durability") == "async"
    assert isinstance(graph.last_kwargs, dict)
    assert graph.last_kwargs.get("durability") == "async"
