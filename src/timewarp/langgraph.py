from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .adapters.installers import bind_langgraph_record
from .adapters.langgraph import LangGraphRecorder
from .determinism import now as tw_now
from .events import Run
from .store import LocalStore


@dataclass
class RecorderHandle:
    """Lightweight facade to record LangGraph runs.

    Constructed via `wrap(graph, ...)`. Call `invoke(inputs, config)` to execute the
    compiled LangGraph while recording events. The resulting run id is exposed as
    `last_run_id` after a successful invocation.
    """

    graph: Any
    store: LocalStore
    project: str
    name: str | None
    labels: dict[str, str]
    privacy_marks: dict[str, str]
    durability: str | None
    stream_modes: Sequence[str]
    snapshot_every: int
    snapshot_on: set[str]
    state_pruner: Callable[[Any], Any] | None
    stream_subgraphs: bool
    require_thread_id: bool
    enable_record_taps: bool

    last_run_id: Any | None = None

    def invoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> Any:
        run = Run(
            project=self.project,
            name=self.name,
            framework="langgraph",
            labels=self.labels,
            started_at=tw_now(),
        )
        teardown: Any | None = None
        recorder = LangGraphRecorder(
            graph=self.graph,
            store=self.store,
            run=run,
            snapshot_every=self.snapshot_every,
            snapshot_on=self.snapshot_on,
            state_pruner=self.state_pruner,
            stream_modes=self.stream_modes,
            stream_subgraphs=self.stream_subgraphs,
            require_thread_id=self.require_thread_id,
            durability=self.durability,
            privacy_marks=self.privacy_marks,
        )
        try:
            if self.enable_record_taps:
                teardown = bind_langgraph_record()
            result = recorder.invoke(inputs, config=config or {})
        finally:
            if callable(teardown):
                try:
                    teardown()
                except Exception:
                    pass
        self.last_run_id = run.run_id
        return result


def wrap(
    graph: Any,
    *,
    project: str,
    name: str | None = None,
    store: LocalStore | None = None,
    labels: dict[str, str] | None = None,
    privacy_marks: dict[str, str] | None = None,
    durability: str | None = None,
    stream_modes: Sequence[str] = ("updates", "messages", "values"),
    snapshot_every: int = 20,
    snapshot_on: Sequence[str] = ("terminal",),
    state_pruner: Callable[[Any], Any] | None = None,
    stream_subgraphs: bool = True,
    require_thread_id: bool = False,
    enable_record_taps: bool = False,
) -> RecorderHandle:
    """Wrap a compiled LangGraph with a recorder facade.

    Parameters
    - graph: Compiled LangGraph.
    - project/name: Run metadata to organize runs.
    - store: Optional LocalStore; defaults to `timewarp.sqlite3` and `blobs/` under CWD.
    - labels: Optional run labels (e.g., {"branch_of": <run_id>}).
    - privacy_marks: Redaction configuration applied at serialization time.
    - durability: Optional stream durability to pass through (e.g., "sync").
    - stream_modes: Which LangGraph stream modes to observe (default: updates+messages+values).
    - snapshot_every: Snapshot cadence in number of update events (default: 20).
    - snapshot_on: Emit snapshots on triggers (e.g., {"terminal","decision"});
      default terminal only.
    - state_pruner: Optional function to prune state payloads before persisting snapshots.
    - stream_subgraphs: Whether to request subgraph streaming (default: True).
    - require_thread_id: Enforce presence of configurable.thread_id in config.
    """

    if store is None:
        store = LocalStore(db_path=Path("timewarp.sqlite3"), blobs_root=Path("blobs"))
    return RecorderHandle(
        graph=graph,
        store=store,
        project=project,
        name=name,
        labels=dict(labels or {}),
        privacy_marks=dict(privacy_marks or {}),
        durability=durability,
        stream_modes=tuple(stream_modes),
        snapshot_every=int(snapshot_every),
        snapshot_on=set(snapshot_on),
        state_pruner=state_pruner,
        stream_subgraphs=bool(stream_subgraphs),
        require_thread_id=bool(require_thread_id),
        enable_record_taps=bool(enable_record_taps),
    )
