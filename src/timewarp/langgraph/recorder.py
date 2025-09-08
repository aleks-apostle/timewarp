from __future__ import annotations

from collections.abc import AsyncIterable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..bindings import try_pop_tool_args_hash as _tw_try_pop_tool_args_hash
from ..codec import to_bytes as _tw_to_bytes
from ..determinism import now as tw_now
from ..determinism import snapshot_rng
from ..events import ActionType, BlobKind, BlobRef, Event, Run, hash_bytes
from ..store import LocalStore
from ..utils.logging import log_warn_once
from .anchors import make_anchor_id as _tw_make_anchor_id
from .batch import EventBatcher
from .classify import ToolClassifier
from .classify import (
    classify_tool_from_update as _tw_classify_tool_from_update,
)
from .classify import (
    default_tool_classifier as _tw_default_tool_classifier,
)
from .classify import (
    infer_action_type as _tw_infer_action_type,
)
from .hashing import extract_tool_args as _tw_extract_tool_args
from .memory import MemoryEmitter
from .memory import (
    infer_mem_scope_from_path as _tw_infer_mem_scope_from_path,
)
from .memory import (
    prune_mem_value as _tw_prune_mem_value,
)
from .messages import MessagesAggregator as _tw_MessagesAggregator
from .retrieval import (
    detect_retrieval as _tw_detect_retrieval,
)
from .retrieval import (
    emit_retrieval_event as _tw_emit_retrieval_event,
)
from .serialize import (
    derive_actor_from_namespace as _tw_derive_actor_from_namespace,
)
from .serialize import (
    extract_checkpoint_id as _tw_extract_checkpoint_id,
)
from .serialize import (
    extract_next_nodes as _tw_extract_next_nodes,
)
from .serialize import (
    extract_values as _tw_extract_values,
)
from .serialize import (
    normalize_bytes as _tw_normalize_bytes,
)
from .serialize import (
    normalize_stream_item as _tw_normalize_stream_item,
)
from .serialize import (
    serialize_messages_tuple as _tw_serialize_messages_tuple,
)
from .stream_async import aiter_stream as _tw_aiter_stream
from .stream_sync import build_stream_kwargs as _tw_build_stream_kwargs
from .stream_sync import iter_stream_sync as _tw_iter_stream_sync
from .taps import flush_provider_taps as _tw_flush_provider_taps
from .versioning import (
    get_timewarp_version as _get_timewarp_version,
)
from .versioning import (
    lib_versions_meta as _lib_versions_meta,
)

"""
Version helpers moved to .versioning to avoid duplication/cycles.
_get_timewarp_version and _lib_versions_meta are imported above.
"""


def _maybe_langgraph_stream(graph: Any) -> bool:
    return hasattr(graph, "stream") and callable(graph.stream)


@dataclass
class LangGraphRecorder:
    """Record LangGraph execution via supported streaming APIs.

    This wrapper observes graph updates and persists Timewarp events. It does not
    mutate the graph; it sits alongside and records events as they occur.
    """

    graph: Any
    store: LocalStore
    run: Run
    snapshot_every: int = 20
    snapshot_on: set[str] = field(default_factory=lambda: {"terminal"})
    state_pruner: Callable[[Any], Any] | None = None
    tool_classifier: ToolClassifier | None = None
    # Memory synthesis controls
    memory_paths: Sequence[str] = ()  # dot paths within values/state to treat as memory
    mem_space_resolver: Callable[[dict[str, str], str], str] | None = None
    memory_pruner: Callable[[Any], Any] | None = None
    # Retrieval detection (optional)
    detect_retrieval: bool = False
    retrieval_pruner: Callable[[Any], Any] | None = None
    # Custom detector can override built-in heuristics; receives values/update payload
    retrieval_detector: Callable[[Any], dict[str, Any] | None] | None = None
    # Defaults per plan: capture updates + values (full state deltas); messages can be opted in
    stream_modes: Sequence[str] = ("updates", "values")
    stream_subgraphs: bool = True
    require_thread_id: bool = False
    durability: str | None = None  # e.g., "sync" | None; pass only when checkpointer present
    privacy_marks: dict[str, str] = field(default_factory=dict)
    # Batch events to reduce SQLite write overhead while preserving order
    event_batch_size: int = 20

    # Adapter metadata (use package version for provenance; keep key compatible)
    ADAPTER_VERSION: str = ""

    def __post_init__(self) -> None:
        # Install a default tool classifier if none was provided
        if self.tool_classifier is None:
            self.tool_classifier = _tw_default_tool_classifier()
        # Normalize adapter version to package version when unset
        if not self.ADAPTER_VERSION:
            self.ADAPTER_VERSION = _get_timewarp_version()
        # Initialize event batcher
        if not hasattr(self, "_batcher"):
            self._batcher = EventBatcher(self.store, self.event_batch_size)
        # Memory emitter encapsulates prior-hash tracking and emission
        if not hasattr(self, "_mem_emitter"):
            self._mem_emitter = MemoryEmitter(
                store=self.store,
                run_id=self.run.run_id,
                adapter_version=(self.ADAPTER_VERSION or _get_timewarp_version()),
                privacy_marks=self.privacy_marks,
                memory_pruner=self.memory_pruner,
            )

    def invoke(self, inputs: dict[str, Any], *, config: dict[str, Any] | None = None) -> Any:
        """Invoke the graph while recording events.

        Requires the compiled graph to support `.stream` with `stream_mode="updates"`.
        """

        if not _maybe_langgraph_stream(self.graph):
            raise RuntimeError("graph does not support .stream; cannot record reliably")

        labels_ctx, thread_id, step = self._prepare_run_and_sys_event(inputs, config)

        # Stream updates and emit events by node update completion.
        # The concrete shape of updates is framework-specific; we record a summary.
        # Determine effective durability again for passing into stream()
        stream_kwargs = _tw_build_stream_kwargs(
            stream_modes=self.stream_modes,
            stream_subgraphs=self.stream_subgraphs,
            durability=self.durability,
            thread_id=thread_id,
        )
        iterator = self._iter_stream_sync(self.graph, inputs, config, stream_kwargs)

        last_values: Any | None = None
        last_decision_key: str | None = None  # track last observed routing decision
        updates_seen = 0  # count LangGraph update chunks to drive snapshot cadence
        single_mode_label: str | None = (
            self.stream_modes[0] if len(self.stream_modes) == 1 else None
        )
        # messages aggregator for this run
        self._msg_agg = _tw_MessagesAggregator(
            store=self.store,
            run_id=self.run.run_id,
            adapter_version=self.ADAPTER_VERSION,
            privacy_marks=self.privacy_marks,
        )

        for update in iterator:
            last_values, last_decision_key, updates_seen, step, thread_id = (
                self._process_one_update(
                    update=update,
                    single_mode_label=single_mode_label,
                    config=config,
                    thread_id=thread_id,
                    agg=None,
                    last_values=last_values,
                    last_decision_key=last_decision_key,
                    updates_seen=updates_seen,
                    step=step,
                )
            )

        # Flush any trailing aggregated messages once stream ends
        try:
            agg = self._msg_agg
        except Exception:
            agg = None
        if agg is not None and agg.has_pending():
            try:
                ev2, step = agg.flush(step=step)
                self._append_event(ev2)
            except Exception as e:  # pragma: no cover - defensive log
                log_warn_once("recorder.flush_messages_failed", e)
        # Final provider-tap flush to capture any late taps
        try:
            _events, step = _tw_flush_provider_taps(
                store=self.store,
                run_id=self.run.run_id,
                step=step,
                actor="graph",
                namespace_label=None,
                thread_id=thread_id,
                adapter_version=self.ADAPTER_VERSION,
                privacy_marks=self.privacy_marks,
                pruner=self.memory_pruner,
            )
            for _ev in _events:
                self._append_event(_ev)
        except Exception as e:  # pragma: no cover
            log_warn_once("recorder.flush_provider_taps_failed", e)

        # Persist a terminal snapshot/state if possible
        terminal_state: Any | None = None
        try:
            # Validate thread_id if required
            if self.require_thread_id:
                th = (
                    (config or {}).get("configurable", {}).get("thread_id")
                    if isinstance(config, dict)
                    else None
                )
                if not th:
                    raise RuntimeError(
                        "thread_id required by recorder but missing in config.configurable"
                    )
            # Prefer graph.get_state(config).values when available
            get_state = getattr(self.graph, "get_state", None)
            if callable(get_state) and config:
                snapshot = get_state(config)
                terminal_state = self._extract_values(snapshot)
        except Exception:
            terminal_state = None

        if terminal_state is not None and "terminal" in (self.snapshot_on or {"terminal"}):
            # Include checkpoint/thread labels on terminal snapshot
            extra_labels2: dict[str, str] = {}
            if thread_id:
                extra_labels2["thread_id"] = thread_id
            cp2 = self._extract_checkpoint_id(config, terminal_state)
            if cp2 is not None:
                extra_labels2["checkpoint_id"] = cp2
            self._persist_snapshot(step, terminal_state, labels_extra=extra_labels2)

        # Flush any pending events before returning (synchronous to avoid cross-thread mutation)
        self._flush_events()
        # Return best-effort result without re-executing the graph
        if last_values is not None:
            return last_values
        if terminal_state is not None:
            return terminal_state
        return None

    # --- async entrypoint ---

    async def ainvoke(self, inputs: dict[str, Any], *, config: dict[str, Any] | None = None) -> Any:
        """Async invoke that mirrors invoke() semantics using graph.astream when available."""

        # Feature detection: must have astream
        has_astream = hasattr(self.graph, "astream") and callable(self.graph.astream)
        if not has_astream:
            raise RuntimeError("graph does not support .astream; cannot record async")

        # Prepare run and initial SYS event
        labels_ctx, thread_id, step = self._prepare_run_and_sys_event(inputs, config)

        # Build stream kwargs mirroring sync
        stream_kwargs = _tw_build_stream_kwargs(
            stream_modes=self.stream_modes,
            stream_subgraphs=self.stream_subgraphs,
            durability=self.durability,
            thread_id=thread_id,
        )

        # Create async iterator, with subgraphs omission resilience
        async_iterator: AsyncIterable[Any] = _tw_aiter_stream(
            self.graph, inputs, config, stream_kwargs
        )

        # Local per-run state, matching sync path
        last_values: Any | None = None
        last_decision_key: str | None = None
        updates_seen = 0
        single_mode_label: str | None = (
            self.stream_modes[0] if len(self.stream_modes) == 1 else None
        )
        # Messages aggregator for async path too
        self._msg_agg = _tw_MessagesAggregator(
            store=self.store,
            run_id=self.run.run_id,
            adapter_version=self.ADAPTER_VERSION,
            privacy_marks=self.privacy_marks,
        )

        async for update in async_iterator:
            last_values, last_decision_key, updates_seen, step, thread_id = (
                self._process_one_update(
                    update=update,
                    single_mode_label=single_mode_label,
                    config=config,
                    thread_id=thread_id,
                    agg=None,
                    last_values=last_values,
                    last_decision_key=last_decision_key,
                    updates_seen=updates_seen,
                    step=step,
                )
            )

        # Flush any trailing aggregated messages once stream ends
        try:
            agg = self._msg_agg
        except Exception:
            agg = None
        if agg is not None and agg.has_pending():
            try:
                ev2, step = agg.flush(step=step)
                self._append_event(ev2)
            except Exception as e:
                log_warn_once("recorder.async.flush_messages_failed", e)

        # Terminal state + optional snapshot
        terminal_state: Any | None = None
        try:
            if self.require_thread_id:
                th = (
                    (config or {}).get("configurable", {}).get("thread_id")
                    if isinstance(config, dict)
                    else None
                )
                if not th:
                    raise RuntimeError(
                        "thread_id required by recorder but missing in config.configurable"
                    )
            get_state = getattr(self.graph, "get_state", None)
            if callable(get_state) and config:
                snapshot = get_state(config)
                terminal_state = self._extract_values(snapshot)
        except Exception as e:
            terminal_state = None
            log_warn_once("recorder.async.get_state_failed", e)

        if terminal_state is not None and "terminal" in (self.snapshot_on or {"terminal"}):
            extra_labels2: dict[str, str] = {}
            if thread_id:
                extra_labels2["thread_id"] = thread_id
            cp2 = self._extract_checkpoint_id(config, terminal_state)
            if cp2 is not None:
                extra_labels2["checkpoint_id"] = cp2
            self._persist_snapshot(step, terminal_state, labels_extra=extra_labels2)

        # Ensure any pending batched events are flushed synchronously to avoid cross-thread mutation
        self._flush_events()

        if last_values is not None:
            return last_values
        if terminal_state is not None:
            return terminal_state
        return None

    # --- helpers ---

    def _persist_snapshot(
        self, step: int, state_like: Any, *, labels_extra: dict[str, str] | None = None
    ) -> None:
        # Apply optional pruner to state payload prior to serialization
        payload = state_like
        if self.state_pruner is not None:
            try:
                pruned = self.state_pruner(state_like)
                # Ensure pruner returns JSON-serializable container; if not, fallback
                if isinstance(pruned, dict | list):
                    payload = pruned
            except Exception:  # pragma: no cover - best-effort
                payload = state_like
        blob = self.store.put_blob(
            self.run.run_id, step, BlobKind.STATE, self._normalize_bytes(payload)
        )
        labs = labels_extra or {}
        ev = Event(
            run_id=self.run.run_id,
            step=step,
            action_type=ActionType.SNAPSHOT,
            actor="graph",
            output_ref=blob,
            hashes={"state": blob.sha256_hex},
            labels=labs,
            model_meta={
                "adapter_version": (self.ADAPTER_VERSION or _get_timewarp_version()),
                "timewarp_version": _get_timewarp_version(),
                "framework": "langgraph",
            },
            ts=tw_now(),
        )
        self._append_event(ev)

    # --- shared helpers for sync/async ---

    def _prepare_run_and_sys_event(
        self, inputs: dict[str, Any], config: dict[str, Any] | None
    ) -> tuple[dict[str, str], str | None, int]:
        """Create run, append the initial SYS event, return (labels, thread_id, next_step)."""
        # Register run
        self.store.create_run(self.run)
        step = 0
        rng_before = snapshot_rng()
        input_blob = self.store.put_blob(
            self.run.run_id, step, BlobKind.INPUT, self._normalize_bytes(inputs)
        )
        # thread id from config
        thread_id: str | None = None
        if isinstance(config, dict):
            cfg = (
                config.get("configurable") if isinstance(config.get("configurable"), dict) else None
            )
            if isinstance(cfg, dict):
                tid = cfg.get("thread_id")
                thread_id = str(tid) if isinstance(tid, str | int) else None

        if self.require_thread_id and not thread_id:
            raise ValueError(
                "require_thread_id=True but no configurable.thread_id provided in config"
            )

        labels: dict[str, str] = {}
        if thread_id:
            labels["thread_id"] = thread_id
        labels["node"] = "graph"
        try:
            if isinstance(self.run.labels, dict) and "branch_of" in self.run.labels:
                bo = self.run.labels.get("branch_of")
                if isinstance(bo, str) and bo:
                    labels["branch_of"] = bo
        except Exception:
            pass
        try:
            if self.stream_modes:
                sm = (
                    ",".join(self.stream_modes)
                    if len(self.stream_modes) > 1
                    else self.stream_modes[0]
                )
                labels["stream_mode"] = str(sm)
            if self.stream_subgraphs:
                labels["subgraphs"] = "true"
            effective_durability = self.durability
            if effective_durability is None and thread_id:
                effective_durability = "sync"
            if effective_durability:
                labels["durability"] = str(effective_durability)
        except Exception:
            pass

        ev = Event(
            run_id=self.run.run_id,
            step=step,
            action_type=ActionType.SYS,
            actor="graph",
            input_ref=input_blob,
            rng_state=rng_before,
            hashes={"input": input_blob.sha256_hex},
            labels=labels,
            model_meta={
                "adapter_version": (self.ADAPTER_VERSION or _get_timewarp_version()),
                "timewarp_version": _get_timewarp_version(),
                "framework": "langgraph",
            },
            ts=tw_now(),
        )
        self._append_event(ev)
        return labels, thread_id, step + 1

    # _build_stream_kwargs moved to stream_sync.build_stream_kwargs

    def _iter_stream_sync(
        self,
        graph: Any,
        inputs: dict[str, Any],
        config: dict[str, Any] | None,
        stream_kwargs: dict[str, Any],
    ) -> Iterable[Any]:
        return _tw_iter_stream_sync(graph, inputs, config, stream_kwargs)

    # Memory synthesis state (path -> last item hash)
    _tw_mem_prev: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def _append_event(self, ev: Event) -> None:
        self._batcher.append(ev)

    def _flush_events(self) -> None:
        self._batcher.flush()

    def _append_events_with_retry(self, events: list[Event]) -> None:
        # Expose for tests that may rely on this method; delegate to batcher
        self._batcher._append_events_with_retry(events)

    # provider taps handled by taps.flush_provider_taps

    def _normalize_bytes(self, obj: Any) -> bytes:
        return _tw_normalize_bytes(
            obj,
            privacy_marks=self.privacy_marks,
            extract_values_fn=self._extract_values,
        )

    # --- memory/tools helpers ---
    def _prune_mem_value(self, value: Any) -> Any:
        return _tw_prune_mem_value(value, self.memory_pruner)

    def _infer_mem_scope_from_path(self, path: str) -> str:
        return _tw_infer_mem_scope_from_path(path)

    # --- retrieval detection & emission ---
    def _detect_retrieval(self, values_like: Any) -> dict[str, Any] | None:
        env = _tw_detect_retrieval(values_like)
        if env is None:
            return None
        # Apply optional overrides/pruning to keep previous behavior
        src = values_like
        try:
            if self.retrieval_detector is not None and isinstance(src, dict):
                custom = self.retrieval_detector(src)
                if isinstance(custom, dict) and custom.get("items"):
                    env = custom
        except Exception:
            pass
        try:
            if self.retrieval_pruner is not None:
                pruned = self.retrieval_pruner(env)
                if isinstance(pruned, dict) and pruned.get("items"):
                    env = pruned
                elif isinstance(pruned, list):
                    env = dict(env)
                    env["items"] = pruned
        except Exception:
            pass
        return env

    def _serialize_messages_tuple(self, pair: Any) -> dict[str, Any]:
        return _tw_serialize_messages_tuple(pair)

    def _normalize_stream_item(
        self, update: Any, single_mode_label: str | None
    ) -> tuple[str | None, str | None, Any]:
        return _tw_normalize_stream_item(update, single_mode_label)

    def _derive_actor_from_namespace(self, namespace_label: str, actor: str) -> str:
        return _tw_derive_actor_from_namespace(namespace_label, actor)

    def _extract_values(self, snapshot_or_obj: Any) -> dict[str, Any] | None:
        out = _tw_extract_values(snapshot_or_obj)
        return out if isinstance(out, dict) else None

    def _extract_next_nodes(self, values_like: Any) -> list[str] | None:
        return _tw_extract_next_nodes(values_like)

    def _extract_checkpoint_id(
        self, config: dict[str, Any] | None, state_or_values: Any
    ) -> str | None:
        return _tw_extract_checkpoint_id(config, state_or_values)

    def _make_anchor_id(
        self,
        action_type: ActionType,
        actor: str,
        labels: dict[str, str],
        tool_name: str | None = None,
    ) -> str:
        return _tw_make_anchor_id(action_type, actor, labels, tool_name)

    # default tool classifier now sourced from classify.default_tool_classifier()
    # internal helper; not part of the public API

    def _process_one_update(
        self,
        *,
        update: Any,
        single_mode_label: str | None,
        config: dict[str, Any] | None,
        thread_id: str | None,
        agg: Any | None,
        last_values: Any | None,
        last_decision_key: str | None,
        updates_seen: int,
        step: int,
    ) -> tuple[Any | None, str | None, int, int, str | None]:
        # Normalize stream item into labels + payload
        namespace_label, mode_label, upd = self._normalize_stream_item(update, single_mode_label)
        stream_mode = mode_label or single_mode_label or "updates"

        # Determine actor from namespace when possible
        actor = "graph"
        if isinstance(namespace_label, str) and namespace_label:
            try:
                actor = self._derive_actor_from_namespace(namespace_label, actor)
            except Exception:
                actor = "graph"

        # Pick up thread_id from payload/metadata when present
        try:
            if isinstance(upd, dict):
                meta = upd.get("metadata") if isinstance(upd.get("metadata"), dict) else None
                th = None
                if meta is not None:
                    th = meta.get("thread_id")
                if th is None:
                    th = upd.get("thread_id")
                if isinstance(th, str) and th:
                    thread_id = th
        except Exception:
            pass

        # Helper to build common labels
        def _mk_labels() -> dict[str, str]:
            labs: dict[str, str] = {}
            try:
                if stream_mode:
                    labs["stream_mode"] = str(stream_mode)
                if namespace_label:
                    labs["namespace"] = str(namespace_label)
                if thread_id:
                    labs["thread_id"] = str(thread_id)
                if actor and actor != "graph":
                    labs["node"] = actor
            except Exception as e:
                log_warn_once("recorder.messages.flush_failed", e)
            return labs

        # Before handling non-messages updates, flush any pending aggregated messages
        try:
            msg_agg = self._msg_agg
        except Exception:
            msg_agg = None

        if stream_mode != "messages" and msg_agg is not None and msg_agg.has_pending():
            ev2, step2 = msg_agg.flush(step=step)
            self._append_event(ev2)
            step = step2

        # Handle messages aggregation
        if stream_mode == "messages":
            try:
                normalized = self._serialize_messages_tuple(upd)
            except Exception:
                normalized = {"message": upd, "metadata": {}}

            if msg_agg is None:
                # Initialize aggregator on-demand (defensive for async path)
                self._msg_agg = _tw_MessagesAggregator(
                    store=self.store,
                    run_id=self.run.run_id,
                    adapter_version=self.ADAPTER_VERSION,
                    privacy_marks=self.privacy_marks,
                )
                msg_agg = self._msg_agg

            # Derive namespace/actor from metadata when available
            try:
                meta = normalized.get("metadata") if isinstance(normalized, dict) else None
                ns2 = None
                if isinstance(meta, dict):
                    ns_val = meta.get("ns")
                    if isinstance(ns_val, list):
                        ns2 = "/".join(str(x) for x in ns_val)
                if ns2:
                    namespace_label = ns2
                    try:
                        actor = self._derive_actor_from_namespace(ns2, actor)
                    except Exception:
                        pass
                if isinstance(meta, dict) and isinstance(meta.get("thread_id"), str):
                    thread_id = str(meta.get("thread_id"))
            except Exception as e:
                log_warn_once("recorder.sys.put_blob_failed", e)

            # Identify aggregation key
            agg_key_new = (actor, namespace_label, thread_id)
            if msg_agg.has_pending() and msg_agg.key() != agg_key_new:
                # Flush previous aggregated chunk as its grouping changed
                ev_prev, step_prev = msg_agg.flush(step=step)
                self._append_event(ev_prev)
                step = step_prev

            if not msg_agg.has_pending():
                msg_agg.start(
                    actor=actor,
                    namespace_label=namespace_label,
                    thread_id=thread_id,
                    normalized=normalized,
                )
            msg_agg.append(normalized)

            # Count the chunk for cadence
            updates_seen += 1

            # Provider taps: best-effort flush after messages chunk
            try:
                _events, step = _tw_flush_provider_taps(
                    store=self.store,
                    run_id=self.run.run_id,
                    step=step,
                    actor=actor,
                    namespace_label=namespace_label,
                    thread_id=thread_id,
                    adapter_version=self.ADAPTER_VERSION,
                    privacy_marks=self.privacy_marks,
                    pruner=self.memory_pruner,
                )
                for _ev in _events:
                    self._append_event(_ev)
            except Exception as e:
                log_warn_once("recorder.values.put_blob_failed", e)

            return last_values, last_decision_key, updates_seen, step, thread_id

        # Handle updates (tool/sys)
        if stream_mode == "updates":
            # Try classify whether this update represents a TOOL action
            at: ActionType
            try:
                at = _tw_infer_action_type(upd)
            except Exception:
                at = ActionType.SYS

            # Handle HITL/interrupt envelopes
            try:
                if isinstance(upd, dict) and "__interrupt__" in upd:
                    labels = _mk_labels()
                    try:
                        aid2 = self._make_anchor_id(ActionType.HITL, actor, labels)
                        if aid2:
                            labels["anchor_id"] = aid2
                    except Exception:
                        pass
                    payload_hitl = {"hitl": {"type": "interrupt", "payload": upd["__interrupt__"]}}
                    out_blob = self.store.put_blob(
                        self.run.run_id, step, BlobKind.OUTPUT, self._normalize_bytes(payload_hitl)
                    )
                    evh = Event(
                        run_id=self.run.run_id,
                        step=step,
                        action_type=ActionType.HITL,
                        actor=actor or "graph",
                        output_ref=out_blob,
                        hashes={"output": out_blob.sha256_hex},
                        labels=labels,
                        model_meta={
                            "adapter_version": (self.ADAPTER_VERSION or _get_timewarp_version()),
                            "timewarp_version": _get_timewarp_version(),
                            "framework": "langgraph",
                        },
                        ts=tw_now(),
                    )
                    self._append_event(evh)
                    step += 1
                    updates_seen += 1
                    # Flush taps after HITL
                    _events, step = _tw_flush_provider_taps(
                        store=self.store,
                        run_id=self.run.run_id,
                        step=step,
                        actor=actor,
                        namespace_label=namespace_label,
                        thread_id=thread_id,
                        adapter_version=self.ADAPTER_VERSION,
                        privacy_marks=self.privacy_marks,
                        pruner=self.memory_pruner,
                    )
                    for _ev in _events:
                        self._append_event(_ev)
                    return last_values, last_decision_key, updates_seen, step, thread_id
            except Exception:
                pass

            if at is ActionType.TOOL:
                labels = _mk_labels()
                # Compute args hash from update shape
                args_hash: str | None = None
                try:
                    norm = _tw_extract_tool_args(upd)
                    if isinstance(norm, dict):
                        args_hash = hash_bytes(_tw_to_bytes(norm))
                except Exception:
                    args_hash = None
                # Merge staged hash (if any) when computed one is missing
                try:
                    if args_hash is None:
                        staged = _tw_try_pop_tool_args_hash()
                        if isinstance(staged, str):
                            args_hash = staged
                            labels["hash_source"] = "staged"
                except Exception:
                    pass

                # Persist a minimal payload for the TOOL event (for context)
                try:
                    blob = self.store.put_blob(
                        self.run.run_id,
                        step,
                        BlobKind.INPUT,
                        self._normalize_bytes({"tool_update": upd}),
                    )
                    hashes: dict[str, str] = {"input": blob.sha256_hex}
                except Exception:
                    blob = None
                    hashes = {}

                if args_hash is not None:
                    hashes["args"] = args_hash

                # Anchor id (tool name if available)
                tool_name: str | None = None
                meta = None
                try:
                    meta = _tw_classify_tool_from_update(upd, self.tool_classifier)
                    tool_name = meta.get("tool_name") if isinstance(meta, dict) else None
                except Exception:
                    meta = None
                try:
                    anchor_id = self._make_anchor_id(ActionType.TOOL, actor, labels, tool_name)
                    if anchor_id:
                        labels["anchor_id"] = anchor_id
                except Exception:
                    pass

                mm_tool = {
                    "adapter_version": (self.ADAPTER_VERSION or _get_timewarp_version()),
                    "timewarp_version": _get_timewarp_version(),
                    "framework": "langgraph",
                }
                try:
                    mm_tool.update(_lib_versions_meta())
                except Exception:
                    pass
                ev = Event(
                    run_id=self.run.run_id,
                    step=step,
                    action_type=ActionType.TOOL,
                    actor=actor or "graph",
                    input_ref=blob,
                    hashes=hashes,
                    labels=labels,
                    model_meta=mm_tool,
                    ts=tw_now(),
                )
                # Attach tool fields on the event model when available
                if isinstance(meta, dict):
                    ev = ev.model_copy(
                        update={
                            "tool_kind": meta.get("tool_kind") or None,
                            "tool_name": meta.get("tool_name") or None,
                            "mcp_server": meta.get("mcp_server") or None,
                            "mcp_transport": meta.get("mcp_transport") or None,
                        }
                    )
                self._append_event(ev)
                step += 1

            elif at is ActionType.LLM:
                labels = _mk_labels()
                # Persist observed update as the LLM output context
                llm_out_blob: BlobRef | None = None
                try:
                    llm_out_blob = self.store.put_blob(
                        self.run.run_id, step, BlobKind.OUTPUT, self._normalize_bytes(upd)
                    )
                except Exception:
                    llm_out_blob = None
                # Extract common meta
                model_meta: dict[str, Any] = {
                    "adapter_version": (self.ADAPTER_VERSION or _get_timewarp_version()),
                    "timewarp_version": _get_timewarp_version(),
                    "framework": "langgraph",
                }
                try:
                    model_meta.update(_lib_versions_meta())
                except Exception:
                    pass
                try:
                    meta = upd.get("metadata") if isinstance(upd, dict) else None
                    if isinstance(meta, dict):
                        for k in ("provider", "model", "temperature", "top_p", "tool_choice"):
                            if k in meta and meta[k] is not None:
                                model_meta[k] = meta[k]
                except Exception:
                    pass
                try:
                    aid = self._make_anchor_id(ActionType.LLM, actor, labels)
                    if aid:
                        labels["anchor_id"] = aid
                except Exception:
                    pass
                llm_hashes: dict[str, str] = {}
                if llm_out_blob is not None:
                    llm_hashes["output"] = llm_out_blob.sha256_hex
                ev_llm = Event(
                    run_id=self.run.run_id,
                    step=step,
                    action_type=ActionType.LLM,
                    actor=actor or "graph",
                    output_ref=llm_out_blob,
                    hashes=llm_hashes,
                    labels=labels,
                    model_meta=model_meta,
                    ts=tw_now(),
                )
                self._append_event(ev_llm)
                step += 1

            else:
                # General SYS event for updates stream
                labels = _mk_labels()
                try:
                    out_blob = self.store.put_blob(
                        self.run.run_id, step, BlobKind.OUTPUT, self._normalize_bytes(upd)
                    )
                    ev = Event(
                        run_id=self.run.run_id,
                        step=step,
                        action_type=ActionType.SYS,
                        actor=actor or "graph",
                        output_ref=out_blob,
                        hashes={},
                        labels=labels,
                        model_meta={
                            "adapter_version": (self.ADAPTER_VERSION or _get_timewarp_version()),
                            "timewarp_version": _get_timewarp_version(),
                            "framework": "langgraph",
                        },
                        ts=tw_now(),
                    )
                    self._append_event(ev)
                    step += 1
                except Exception:
                    pass

            # Count update for cadence and flush provider taps
            updates_seen += 1
            try:
                _events, step = _tw_flush_provider_taps(
                    store=self.store,
                    run_id=self.run.run_id,
                    step=step,
                    actor=actor,
                    namespace_label=namespace_label,
                    thread_id=thread_id,
                    adapter_version=self.ADAPTER_VERSION,
                    privacy_marks=self.privacy_marks,
                    pruner=self.memory_pruner,
                )
                for _ev in _events:
                    self._append_event(_ev)
            except Exception:
                pass

            return last_values, last_decision_key, updates_seen, step, thread_id

        # Handle values stream
        if stream_mode == "values":
            labels = _mk_labels()

            # Choose a state-like payload to persist and derive last_values
            state_like = upd
            try:
                # Persist the chunk as observed for transparency; do not hash to avoid
                # nondeterministic diffs across parallel updates ordering.
                out_blob = self.store.put_blob(
                    self.run.run_id, step, BlobKind.OUTPUT, self._normalize_bytes(state_like)
                )
                ev = Event(
                    run_id=self.run.run_id,
                    step=step,
                    action_type=ActionType.SYS,
                    actor=actor or "graph",
                    output_ref=out_blob,
                    hashes={},
                    labels=labels,
                    model_meta={
                        "adapter_version": (self.ADAPTER_VERSION or _get_timewarp_version()),
                        "timewarp_version": _get_timewarp_version(),
                        "framework": "langgraph",
                    },
                    ts=tw_now(),
                )
                self._append_event(ev)
                step += 1
            except Exception:
                pass

            # Update last_values from typical shapes
            try:
                vals = self._extract_values(state_like)
                if isinstance(vals, dict):
                    last_values = vals.get("state") if isinstance(vals.get("state"), dict) else vals
                elif isinstance(state_like, dict):
                    last_values = (
                        state_like.get("state")
                        if isinstance(state_like.get("state"), dict)
                        else state_like
                    )
            except Exception:
                pass

            # Emit DECISION event when routing changes
            try:
                next_nodes = self._extract_next_nodes(state_like)
                decision_key = ",".join(next_nodes) if isinstance(next_nodes, list) else None
            except Exception as e:
                decision_key = None
                log_warn_once("recorder.values.extract_next_nodes_failed", e)
            if decision_key and decision_key != last_decision_key:
                labs2 = _mk_labels()
                labs2["decision"] = decision_key
                # Anchor id for decisions as well
                try:
                    aid = self._make_anchor_id(ActionType.DECISION, actor, labs2)
                    if aid:
                        labs2["anchor_id"] = aid
                except Exception as e:
                    log_warn_once("recorder.values.snapshot_after_decision_failed", e)
                evd = Event(
                    run_id=self.run.run_id,
                    step=step,
                    action_type=ActionType.DECISION,
                    actor=actor or "graph",
                    hashes={},
                    labels=labs2,
                    model_meta={
                        "adapter_version": (self.ADAPTER_VERSION or _get_timewarp_version()),
                        "timewarp_version": _get_timewarp_version(),
                        "framework": "langgraph",
                    },
                    ts=tw_now(),
                )
                self._append_event(evd)
                step += 1
                last_decision_key = decision_key

                # Snapshot immediately after decision if enabled
                try:
                    if "decision" in (self.snapshot_on or set()):
                        self._persist_snapshot(step, state_like, labels_extra=_mk_labels())
                        step += 1
                except Exception:
                    pass

            # Snapshot by cadence
            try:
                updates_seen += 1
                se = int(self.snapshot_every)
                if se > 0 and (updates_seen % se) == 0:
                    self._persist_snapshot(step, state_like, labels_extra=_mk_labels())
                    step += 1
            except Exception as e:
                log_warn_once("recorder.values.snapshot_by_cadence_failed", e)

            # Memory synthesis from configured value paths
            try:
                if self.memory_paths:
                    vals_for_mem = self._extract_values(state_like)
                    if isinstance(vals_for_mem, dict):
                        step, mem_events = self._mem_emitter.emit_from_values(
                            step=step,
                            actor=actor,
                            namespace_label=namespace_label,
                            thread_id=thread_id,
                            values=vals_for_mem,
                            memory_paths=tuple(self.memory_paths),
                            mem_space_resolver=self.mem_space_resolver,
                        )
                        for _ev3 in mem_events:
                            self._append_event(_ev3)
            except Exception as e:
                log_warn_once("recorder.values.memory_emit_failed", e)

            # Retrieval detection (values-based)
            try:
                if self.detect_retrieval:
                    env = self._detect_retrieval(state_like)
                    if isinstance(env, dict) and env.get("items"):
                        evr, step2 = _tw_emit_retrieval_event(
                            store=self.store,
                            run_id=self.run.run_id,
                            step=step,
                            actor=actor,
                            namespace_label=namespace_label,
                            thread_id=thread_id,
                            env=env,
                            adapter_version=self.ADAPTER_VERSION,
                            privacy_marks=self.privacy_marks,
                        )
                        if isinstance(evr, Event):
                            self._append_event(evr)
                            step = step2
            except Exception as e:
                log_warn_once("recorder.values.retrieval_emit_failed", e)

            # Provider taps flush after values chunk
            try:
                _events, step = _tw_flush_provider_taps(
                    store=self.store,
                    run_id=self.run.run_id,
                    step=step,
                    actor=actor,
                    namespace_label=namespace_label,
                    thread_id=thread_id,
                    adapter_version=self.ADAPTER_VERSION,
                    privacy_marks=self.privacy_marks,
                    pruner=self.memory_pruner,
                )
                for _ev4 in _events:
                    self._append_event(_ev4)
            except Exception as e:
                log_warn_once("recorder.values.flush_provider_taps_failed", e)

            return last_values, last_decision_key, updates_seen, step, thread_id

        # Fallback: treat as a plain SYS chunk
        try:
            labels = _mk_labels()
            out_blob = self.store.put_blob(
                self.run.run_id, step, BlobKind.OUTPUT, self._normalize_bytes(upd)
            )
            ev = Event(
                run_id=self.run.run_id,
                step=step,
                action_type=ActionType.SYS,
                actor=actor or "graph",
                output_ref=out_blob,
                hashes={"output": out_blob.sha256_hex},
                labels=labels,
                model_meta={
                    "adapter_version": (self.ADAPTER_VERSION or _get_timewarp_version()),
                    "timewarp_version": _get_timewarp_version(),
                    "framework": "langgraph",
                },
                ts=tw_now(),
            )
            self._append_event(ev)
            step += 1
            updates_seen += 1
        except Exception as e:
            log_warn_once("recorder.fallback.put_blob_failed", e)

        return last_values, last_decision_key, updates_seen, step, thread_id
