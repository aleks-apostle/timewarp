from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..bindings import (
    begin_recording_session,
    bind_langgraph_record,
    bind_memory_taps,
)
from ..determinism import now as tw_now
from ..events import Run
from ..store import LocalStore
from ..utils.fingerprint import runtime_labels as _tw_runtime_labels
from ..utils.logging import log_warn_once
from .classify import ToolClassifier
from .recorder import LangGraphRecorder


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
    enable_memory_taps: bool
    tool_classifier: ToolClassifier | None = None
    retrieval_detector: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None
    event_batch_size: int = 20

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
        teardown_mem: Any | None = None
        end_session: Any | None = None
        recorder = LangGraphRecorder(
            graph=self.graph,
            store=self.store,
            run=run,
            snapshot_every=self.snapshot_every,
            snapshot_on=self.snapshot_on,
            state_pruner=self.state_pruner,
            tool_classifier=self.tool_classifier,
            retrieval_detector=self.retrieval_detector,
            stream_modes=self.stream_modes,
            stream_subgraphs=self.stream_subgraphs,
            require_thread_id=self.require_thread_id,
            durability=self.durability,
            privacy_marks=self.privacy_marks,
            event_batch_size=self.event_batch_size,
        )
        try:
            # Scope staged hashes/memory taps to this run
            end_session = begin_recording_session(run.run_id)
            if self.enable_record_taps:
                teardown = bind_langgraph_record()
            if self.enable_memory_taps:
                teardown_mem = bind_memory_taps()
            result = recorder.invoke(inputs, config=config or {})
        finally:
            if callable(end_session):
                try:
                    end_session()
                except Exception as e:
                    log_warn_once("langgraph.session.reset_failed", e)
            if callable(teardown):
                try:
                    teardown()
                except Exception as e:
                    log_warn_once("langgraph.teardown.record_taps_failed", e)
            if callable(teardown_mem):
                try:
                    teardown_mem()
                except Exception as e:
                    log_warn_once("langgraph.teardown.memory_taps_failed", e)
        self.last_run_id = run.run_id
        return result

    async def ainvoke(self, inputs: dict[str, Any], config: dict[str, Any] | None = None) -> Any:
        run = Run(
            project=self.project,
            name=self.name,
            framework="langgraph",
            labels=self.labels,
            started_at=tw_now(),
        )
        teardown: Any | None = None
        teardown_mem: Any | None = None
        end_session: Any | None = None
        recorder = LangGraphRecorder(
            graph=self.graph,
            store=self.store,
            run=run,
            snapshot_every=self.snapshot_every,
            snapshot_on=self.snapshot_on,
            state_pruner=self.state_pruner,
            tool_classifier=self.tool_classifier,
            retrieval_detector=self.retrieval_detector,
            stream_modes=self.stream_modes,
            stream_subgraphs=self.stream_subgraphs,
            require_thread_id=self.require_thread_id,
            durability=self.durability,
            privacy_marks=self.privacy_marks,
            event_batch_size=self.event_batch_size,
        )
        try:
            # Scope staged hashes/memory taps to this run
            end_session = begin_recording_session(run.run_id)
            if self.enable_record_taps:
                teardown = bind_langgraph_record()
            if self.enable_memory_taps:
                teardown_mem = bind_memory_taps()
            # Prefer native async recording
            result = await recorder.ainvoke(inputs, config=config or {})
        finally:
            if callable(end_session):
                try:
                    # Reset session context in a thread to avoid blocking loop if needed
                    await asyncio.to_thread(end_session)
                except Exception as e:
                    log_warn_once("langgraph.async.session.reset_failed", e)
            if callable(teardown):
                try:
                    # Teardown is currently sync best-effort; run it in a thread to avoid blocking
                    await asyncio.to_thread(teardown)
                except Exception as e:
                    log_warn_once("langgraph.async.teardown.record_taps_failed", e)
            if callable(teardown_mem):
                try:
                    await asyncio.to_thread(teardown_mem)
                except Exception as e:
                    log_warn_once("langgraph.async.teardown.memory_taps_failed", e)
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
    stream_modes: Sequence[str] = ("updates", "values"),
    snapshot_every: int = 20,
    snapshot_on: Sequence[str] = ("terminal",),
    state_pruner: Callable[[Any], Any] | None = None,
    tool_classifier: ToolClassifier | None = None,
    retrieval_detector: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    stream_subgraphs: bool = True,
    require_thread_id: bool = False,
    enable_record_taps: bool = True,
    enable_memory_taps: bool = True,
    event_batch_size: int = 20,
    busy_timeout_ms: int | None = 5000,
) -> RecorderHandle:
    """Wrap a compiled LangGraph with a recorder facade.

    Parameters
    - graph: Compiled LangGraph.
    - project/name: Run metadata to organize runs.
    - store: Optional LocalStore; defaults to `timewarp.sqlite3` and `blobs/` under CWD.
    - labels: Optional run labels (e.g., {"branch_of": <run_id>}).
    - privacy_marks: Redaction configuration applied at serialization time.
    - durability: Optional stream durability to pass through (e.g., "sync").
    - stream_modes: Which LangGraph stream modes to observe
      (default: updates+values; messages opt-in).
    - snapshot_every: Snapshot cadence in number of update events (default: 20).
    - snapshot_on: Emit snapshots on triggers (e.g., {"terminal","decision"});
      default terminal only.
    - state_pruner: Optional function to prune state payloads before persisting snapshots.
    - stream_subgraphs: Whether to request subgraph streaming (default: True).
    - require_thread_id: Enforce presence of configurable.thread_id in config.
    - event_batch_size: Buffer and batch-append events (default: 20) for throughput.
    """

    if store is None:
        store = LocalStore(
            db_path=Path("timewarp.sqlite3"),
            blobs_root=Path("blobs"),
            busy_timeout_ms=busy_timeout_ms,
        )
    # Allow environment override for record taps: TIMEWARP_RECORD_TAPS=0 disables
    eff_enable_taps = enable_record_taps
    try:
        import os as _os  # local import to keep core deterministic logic separate

        val = _os.environ.get("TIMEWARP_RECORD_TAPS")
        if val is not None:
            v = val.strip().lower()
            if v in {"0", "false", "no", "off"}:
                eff_enable_taps = False
            elif v in {"1", "true", "yes", "on"}:
                eff_enable_taps = True
    except Exception:
        # best-effort; env parsing should never break
        pass

    # Merge runtime fingerprint labels without overriding explicit caller labels
    labels2 = dict(labels or {})
    try:
        fps = _tw_runtime_labels()
        for k, v in fps.items():
            labels2.setdefault(k, v)
    except Exception:
        pass

    return RecorderHandle(
        graph=graph,
        store=store,
        project=project,
        name=name,
        labels=labels2,
        privacy_marks=dict(privacy_marks or {}),
        durability=durability,
        stream_modes=tuple(stream_modes),
        snapshot_every=int(snapshot_every),
        snapshot_on=set(snapshot_on),
        state_pruner=state_pruner,
        tool_classifier=tool_classifier,
        retrieval_detector=retrieval_detector,
        stream_subgraphs=bool(stream_subgraphs),
        require_thread_id=bool(require_thread_id),
        enable_record_taps=bool(eff_enable_taps),
        enable_memory_taps=bool(enable_memory_taps),
        event_batch_size=int(event_batch_size),
    )


__all__ = [
    "LangGraphRecorder",
    "RecorderHandle",
    "wrap",
]
