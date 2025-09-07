from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from timewarp.events import Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _NoSubgraphsParamGraph:
    """Graph whose stream() signature does not accept `subgraphs` kwarg.

    Recorder should retry without `subgraphs` when TypeError occurs.
    """

    def stream(
        self,
        inputs: dict[str, Any],
        config: dict[str, Any] | None = None,
        *,
        stream_mode: list[str] | str,
        durability: str | None = None,
    ) -> Iterable[Any]:
        # Simple one-chunk stream
        _ = stream_mode, durability
        yield ("updates", {"ok": True})


def test_stream_retry_without_subgraphs_kwarg(tmp_path: Path) -> None:
    graph = _NoSubgraphsParamGraph()
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="nosg", framework="langgraph")

    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        stream_subgraphs=True,  # will be dropped on retry
    )
    _ = rec.invoke({"x": 1}, config={})

    # Ensure we recorded at least the initial SYS + one update event
    events = store.list_events(run.run_id)
    assert len(events) >= 2
