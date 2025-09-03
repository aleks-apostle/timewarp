from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from ..determinism import now as tw_now
from ..determinism import snapshot_rng
from ..events import ActionType, BlobKind, Event, Run, hash_bytes
from ..store import LocalStore
from .installers import try_pop_prompt_hash, try_pop_tool_args_hash


class ToolClassifier(Protocol):
    def __call__(self, tool: Any) -> dict[str, str] | None:
        """Return MCP/tool metadata if tool is MCP, else None.

        Expected keys when MCP: {"tool_kind": "MCP", "tool_name": str,
        "mcp_server": str, "mcp_transport": str}
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
    memory_keys: Sequence[str] = ()  # dot paths within values/state to treat as memory
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
    event_batch_size: int = 1

    # Adapter metadata
    ADAPTER_VERSION: str = "0.1.0"

    def __post_init__(self) -> None:
        # Install a default tool classifier if none was provided
        if self.tool_classifier is None:
            self.tool_classifier = _default_tool_classifier()

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
        stream_kwargs = self._build_stream_kwargs(config, thread_id)
        iterator = self._iter_stream_sync(self.graph, inputs, config, stream_kwargs)

        last_values: Any | None = None
        last_decision_key: str | None = None  # track last observed routing decision
        updates_seen = 0  # count LangGraph update chunks to drive snapshot cadence
        single_mode_label: str | None = None
        if len(self.stream_modes) == 1:
            single_mode_label = self.stream_modes[0]
        # Simple aggregation buffer for messages stream: one LLM event per node/namespace
        agg_key: tuple[str, str | None, str | None] | None = None
        agg_chunks: list[dict[str, Any]] = []
        agg_text: list[str] = []
        agg_labels: dict[str, str] = {}
        agg_actor: str | None = None

        def flush_messages() -> None:
            nonlocal step, agg_key, agg_chunks, agg_text, agg_labels, agg_actor
            if agg_key is None:
                return
            ev2, step2 = self._finalize_messages_aggregate(
                step=step,
                agg_actor=agg_actor,
                agg_text=agg_text,
                agg_chunks=agg_chunks,
                agg_labels=agg_labels,
            )
            self._append_event(ev2)
            step = step2
            # reset
            agg_key = None
            agg_chunks = []
            agg_text = []
            agg_labels = {}
            agg_actor = None

        for update in iterator:
            # Normalize stream shapes
            namespace_label, stream_mode_label, upd = self._normalize_stream_item(
                update, single_mode_label
            )

            actor = self._infer_actor(upd)
            # Derive actor from namespace if not inferred
            if actor == "graph" and namespace_label:
                actor = self._derive_actor_from_namespace(namespace_label, actor)

            # Special handling for messages mode: (message_chunk, metadata)
            if (
                (stream_mode_label == "messages" or single_mode_label == "messages")
                and isinstance(upd, tuple | list)
                and len(upd) == 2
            ):
                # Capture labels from metadata if present before serialization
                try:
                    meta_obj = upd[1]
                    if hasattr(meta_obj, "get"):
                        # defer setting langgraph_node until after normalization
                        # ns in metadata as list
                        ns = meta_obj.get("ns")
                        if ns and not namespace_label:
                            try:
                                namespace_label = "/".join([str(x) for x in ns])
                            except Exception:
                                pass
                        # thread id hint
                        tid = meta_obj.get("thread_id")
                        if tid and isinstance(tid, str | int) and not thread_id:
                            thread_id = str(tid)
                    # no-op if meta doesn't have mapping methods
                except Exception:
                    pass
                # Aggregate messages per (actor, namespace, thread) until boundary
                normalized = self._serialize_messages_tuple(upd)
                # Determine aggregation key
                key = (actor, namespace_label, thread_id)
                if agg_key is None:
                    agg_key = key
                    agg_labels = {}
                    if namespace_label:
                        agg_labels["namespace"] = namespace_label
                    if thread_id:
                        agg_labels["thread_id"] = thread_id
                    # langgraph node label if available
                    try:
                        if isinstance(normalized.get("metadata"), dict):
                            ln = normalized["metadata"].get("langgraph_node")
                            if ln:
                                agg_labels["node"] = str(ln)
                    except Exception:
                        pass
                    agg_actor = actor
                # If key changes, flush and start new aggregate
                elif agg_key != key:
                    flush_messages()
                    agg_key = key
                    agg_labels = {}
                    if namespace_label:
                        agg_labels["namespace"] = namespace_label
                    if thread_id:
                        agg_labels["thread_id"] = thread_id
                    try:
                        if isinstance(normalized.get("metadata"), dict):
                            ln2 = normalized["metadata"].get("langgraph_node")
                            if ln2:
                                agg_labels["node"] = str(ln2)
                    except Exception:
                        pass
                    agg_actor = actor
                    agg_chunks = []
                    agg_text = []
                # Append chunk content
                agg_chunks.append(normalized)
                try:
                    msg = normalized.get("message")
                    # message is likely a dict with content
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        agg_text.append(msg["content"])
                    elif isinstance(msg, str):
                        agg_text.append(msg)
                except Exception:
                    pass
                # Defer event creation; continue to next update
                continue

            # On encountering a non-messages update, flush any pending messages aggregate first
            flush_messages()
            payload_b = self._normalize_bytes(upd)
            out_blob = self.store.put_blob(self.run.run_id, step, BlobKind.OUTPUT, payload_b)
            ev_labels: dict[str, str] = {}
            if stream_mode_label:
                ev_labels["stream_mode"] = stream_mode_label
            if namespace_label:
                ev_labels["namespace"] = namespace_label
            if thread_id:
                ev_labels.setdefault("thread_id", thread_id)
            # For non-messages updates, set langgraph_node label from actor if known
            if actor and actor != "graph":
                ev_labels.setdefault("node", actor)

            # Set action type
            atype = self._infer_action_type(upd)

            # adapter meta baseline; merge with any event-specific meta
            base_meta = {"adapter_version": self.ADAPTER_VERSION, "framework": "langgraph"}
            ev_meta: dict[str, Any] | None = None
            # Set action type; force LLM for messages-mode streams
            atype = self._infer_action_type(upd)
            if stream_mode_label == "messages" or single_mode_label == "messages":
                atype = ActionType.LLM

            # Tool classification (MCP metadata, etc.) and force TOOL type when recognized
            tool_meta: dict[str, str] | None = self._classify_tool_from_update(upd)
            if tool_meta:
                atype = ActionType.TOOL

            # Include specific meta for messages aggregation if present
            if (
                atype is ActionType.LLM
                and isinstance(upd, dict)
                and isinstance(upd.get("metadata"), dict)
                and "chunks_count" in upd.get("metadata", {})
            ):
                # When messages are already aggregated in payload (rare here), pick count
                try:
                    ev_meta = {"chunks_count": int(upd["metadata"]["chunks_count"])}
                except Exception:
                    ev_meta = {"chunks_count": 1}

            # Build hashes with optional prompt hash for LLM
            ev_hashes: dict[str, str] = {"output": out_blob.sha256_hex}
            input_ref = None
            ev_tools_digest: str | None = None
            if atype is ActionType.LLM:
                try:
                    from ..codec import to_bytes as _to_bytes

                    cand = None
                    obs = upd
                    if isinstance(upd, dict) and len(upd) == 1:
                        try:
                            ((_, inner),) = upd.items()
                            if isinstance(inner, dict):
                                obs = inner
                        except Exception:
                            obs = upd
                    if isinstance(obs, dict):
                        for pkey in ("llm_input_messages", "input_messages", "messages", "prompt"):
                            if pkey in obs:
                                cand = obs[pkey]
                                break
                    if cand is not None:
                        ev_hashes["prompt"] = hash_bytes(_to_bytes(cand))
                        # Label hash provenance for non-messages LLM event
                        try:
                            ev_labels["hash_source"] = "aggregated"
                        except Exception:
                            pass
                except Exception:
                    pass
                # Tools digest and prompt context (async non-messages)
                try:
                    tools = self._extract_tools_from_update(upd)
                    if tools is not None:
                        from ..codec import to_bytes as _to_bytes
                        from ..events import hash_bytes as _hash

                        td = _hash(_to_bytes({"tools": tools}))
                        ev_hashes["tools"] = td
                        ctx_messages = None
                        try:
                            obs2 = upd if isinstance(upd, dict) else {}
                            for pkey in (
                                "llm_input_messages",
                                "input_messages",
                                "messages",
                                "prompt",
                            ):
                                if isinstance(obs2, dict) and pkey in obs2:
                                    ctx_messages = obs2[pkey]
                                    break
                        except Exception:
                            ctx_messages = None
                        if ctx_messages is not None:
                            ctx_obj = {"messages": ctx_messages, "tools": tools}
                            parts_b = self._normalize_bytes(ctx_obj)
                            input_ref = self.store.put_blob(
                                self.run.run_id, step, BlobKind.INPUT, parts_b
                            )
                            ev_hashes["prompt_ctx"] = _hash(_to_bytes(ctx_obj))
                        ev_tools_digest = td
                except Exception:
                    pass
                # Merge staged prompt hash when stream metadata is insufficient
                try:
                    if "prompt" not in ev_hashes:
                        staged_p = try_pop_prompt_hash()
                        if staged_p:
                            ev_hashes["prompt"] = staged_p
                            try:
                                # Staged provenance takes precedence
                                ev_labels["hash_source"] = "staged"
                            except Exception:
                                pass
                except Exception:
                    pass
                # Extract common LLM params from metadata when present (best-effort)
                try:
                    meta_obj = upd.get("metadata") if isinstance(upd, dict) else None
                    if isinstance(meta_obj, dict):
                        # Provider/model when present
                        try:
                            prov = meta_obj.get("provider")
                            mdl = meta_obj.get("model")
                            if prov is not None:
                                if ev_meta is None:
                                    ev_meta = {}
                                if isinstance(prov, str):
                                    ev_meta["provider"] = prov
                            if mdl is not None:
                                if ev_meta is None:
                                    ev_meta = {}
                                if isinstance(mdl, str):
                                    ev_meta["model"] = mdl
                        except Exception:
                            pass
                        for p in ("temperature", "top_p", "tool_choice"):
                            val = meta_obj.get(p)
                            if val is None and isinstance(meta_obj.get("params"), dict):
                                params2 = meta_obj["params"]
                                try:
                                    val = params2.get(p)
                                except Exception:
                                    val = None
                            if val is not None:
                                if ev_meta is None:
                                    ev_meta = {}
                                # keep only JSON-serializable scalars
                                if isinstance(val, str | int | float | bool):
                                    ev_meta[p] = val
                except Exception:
                    pass
                # Optional: tools digest + prompt context blob
                try:
                    tools = self._extract_tools_from_update(upd)
                    if tools is not None:
                        from ..codec import to_bytes as _to_bytes
                        from ..events import hash_bytes as _hash

                        td = _hash(_to_bytes({"tools": tools}))
                        ev_hashes["tools"] = td
                        # prompt context combines messages and tools when messages are available
                        ctx_messages = None
                        try:
                            obs2 = upd if isinstance(upd, dict) else {}
                            for pkey in (
                                "llm_input_messages",
                                "input_messages",
                                "messages",
                                "prompt",
                            ):
                                if isinstance(obs2, dict) and pkey in obs2:
                                    ctx_messages = obs2[pkey]
                                    break
                        except Exception:
                            ctx_messages = None
                        if ctx_messages is not None:
                            ctx_obj = {"messages": ctx_messages, "tools": tools}
                            parts_b = self._normalize_bytes(ctx_obj)
                            input_ref = self.store.put_blob(
                                self.run.run_id, step, BlobKind.INPUT, parts_b
                            )
                            ev_hashes["prompt_ctx"] = _hash(_to_bytes(ctx_obj))
                        ev_tools_digest = td
                except Exception:
                    pass
            # Build tool args hash when actionable
            if atype is ActionType.TOOL:
                try:
                    from ..codec import to_bytes as _to_bytes

                    args_env = self._extract_tool_args(upd)
                    if args_env is not None:
                        ev_hashes["args"] = hash_bytes(_to_bytes(args_env))
                except Exception:
                    pass
                # Merge staged tool args hash when not present in update payload
                try:
                    if "args" not in ev_hashes:
                        staged_a = try_pop_tool_args_hash()
                        if staged_a:
                            ev_hashes["args"] = staged_a
                except Exception:
                    pass

            # Compute anchor id for alignment and diffing
            try:
                tool_nm = None
                if isinstance(upd, dict):
                    tool_nm = str(upd.get("tool_name")) if upd.get("tool_name") else None
                anchor_id = self._make_anchor_id(atype, actor, ev_labels, tool_nm)
                if anchor_id:
                    ev_labels["anchor_id"] = anchor_id
            except Exception:
                pass

            ev = Event(
                run_id=self.run.run_id,
                step=step,
                action_type=atype,
                actor=actor,
                input_ref=(
                    input_ref if atype is ActionType.LLM and input_ref is not None else None
                ),
                output_ref=out_blob,
                hashes=ev_hashes,
                labels=ev_labels,
                model_meta=(base_meta if ev_meta is None else {**base_meta, **ev_meta}),
                ts=tw_now(),
            )
            if tool_meta:
                ev = ev.model_copy(update=tool_meta)
            if atype is ActionType.LLM and ev_tools_digest is not None:
                ev = ev.model_copy(update={"tools_digest": ev_tools_digest})

            self._append_event(ev)
            step += 1  # advance step after recording the update event
            updates_seen += 1
            # Emit MEMORY events synthesized from values stream (when configured)
            try:
                if (stream_mode_label == "values" or single_mode_label == "values") and isinstance(
                    upd, dict
                ):
                    vals = self._extract_values(upd)
                    if isinstance(vals, dict):
                        if self.memory_keys:
                            step = self._emit_memory_events(
                                step=step,
                                actor=actor,
                                namespace_label=namespace_label,
                                thread_id=thread_id,
                                values=vals,
                            )
                        # Optional retrieval detection from values
                        if self.detect_retrieval:
                            env = self._detect_retrieval(vals)
                            if isinstance(env, dict):
                                step = self._emit_retrieval_event(
                                    step=step,
                                    actor=actor,
                                    namespace_label=namespace_label,
                                    thread_id=thread_id,
                                    env=env,
                                )
            except Exception:  # pragma: no cover
                pass
            # DECISION emission when values.next changes
            try:
                if (stream_mode_label == "values" or single_mode_label == "values") and isinstance(
                    upd, dict
                ):
                    next_nodes = self._extract_next_nodes(upd)
                    if next_nodes is not None:
                        decision_key = "|".join(next_nodes)
                        if decision_key != last_decision_key:
                            dec_labels: dict[str, str] = {}
                            if namespace_label:
                                dec_labels["namespace"] = namespace_label
                            if thread_id:
                                dec_labels["thread_id"] = thread_id
                            if actor and actor != "graph":
                                dec_labels["node"] = actor
                            # Anchor for DECISION
                            try:
                                dec_anchor = self._make_anchor_id(
                                    ActionType.DECISION, actor, dec_labels
                                )
                                if dec_anchor:
                                    dec_labels["anchor_id"] = dec_anchor
                            except Exception:
                                pass
                            dec_labels["decision"] = decision_key
                            dec_ev = Event(
                                run_id=self.run.run_id,
                                step=step,
                                action_type=ActionType.DECISION,
                                actor=actor or "router",
                                labels=dec_labels,
                                model_meta={
                                    "adapter_version": self.ADAPTER_VERSION,
                                    "framework": "langgraph",
                                },
                                ts=tw_now(),
                            )
                            self._append_event(dec_ev)
                            step += 1
                            last_decision_key = decision_key
                            # Optional snapshot on decision
                            try:
                                if "decision" in self.snapshot_on:
                                    snap_payload_dec: Any | None = None
                                    if last_values is not None and isinstance(
                                        last_values, dict | list
                                    ):
                                        snap_payload_dec = last_values
                                    else:
                                        get_state = getattr(self.graph, "get_state", None)
                                        if callable(get_state) and config:
                                            snapshot = get_state(config)
                                            snap_payload_dec = self._extract_values(snapshot)
                                    if snap_payload_dec is not None:
                                        extra_labels_dec: dict[str, str] = {}
                                        if thread_id:
                                            extra_labels_dec["thread_id"] = thread_id
                                        cpd = self._extract_checkpoint_id(config, snap_payload_dec)
                                        if cpd is not None:
                                            extra_labels_dec["checkpoint_id"] = cpd
                                        self._persist_snapshot(
                                            step, snap_payload_dec, labels_extra=extra_labels_dec
                                        )
                                        step += 1
                            except Exception:  # pragma: no cover - best-effort
                                pass
            except Exception:  # pragma: no cover - best-effort
                pass
            # Track final state when using values mode (do this before snapshotting)
            try:
                if (stream_mode_label == "values" or single_mode_label == "values") and isinstance(
                    upd, dict
                ):
                    last_values = upd
            except Exception:  # pragma: no cover - best-effort
                pass
            if self.snapshot_every > 0 and updates_seen % self.snapshot_every == 0:
                # Prefer a full state snapshot from values stream or graph.get_state when possible
                snap_payload: Any | None = None
                if last_values is not None and isinstance(last_values, dict | list):
                    snap_payload = last_values
                else:
                    try:
                        get_state = getattr(self.graph, "get_state", None)
                        if callable(get_state) and config:
                            snapshot = get_state(config)
                            snap_payload = self._extract_values(snapshot)
                    except Exception:
                        snap_payload = None
                if snap_payload is not None:
                    # Attach checkpoint_id label if available
                    extra_labels: dict[str, str] = {}
                    if thread_id:
                        extra_labels["thread_id"] = thread_id
                    cp = self._extract_checkpoint_id(config, snap_payload)
                    if cp is not None:
                        extra_labels["checkpoint_id"] = cp
                    self._persist_snapshot(step, snap_payload, labels_extra=extra_labels)
                    step += 1

            # HITL detection best-effort directly from updates
            try:
                if isinstance(upd, dict) and "__interrupt__" in upd:
                    # Normalize into explicit envelope to make HITL events consistent
                    _hitl_payload = {"hitl": {"type": "interrupt", "payload": upd["__interrupt__"]}}
                    hitl_blob = self.store.put_blob(
                        self.run.run_id,
                        step,
                        BlobKind.OUTPUT,
                        self._normalize_bytes(_hitl_payload),
                    )
                    hitl_ev = Event(
                        run_id=self.run.run_id,
                        step=step,
                        action_type=ActionType.HITL,
                        actor=actor,
                        output_ref=hitl_blob,
                        hashes={"output": hitl_blob.sha256_hex},
                        labels=(ev_labels if ev_labels else {}),
                        model_meta={
                            "adapter_version": self.ADAPTER_VERSION,
                            "framework": "langgraph",
                        },
                        ts=tw_now(),
                    )
                    self._append_event(hitl_ev)
                    step += 1
            except Exception:  # pragma: no cover
                pass

        # Flush any trailing aggregated messages once stream ends
        flush_messages()

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

        # Flush any pending events before returning
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
        stream_kwargs = self._build_stream_kwargs(config, thread_id)

        # Create async iterator, with subgraphs omission resilience
        try:
            async_iterator: AsyncIterable[Any] = cast(
                AsyncIterable[Any], self.graph.astream(inputs, config or {}, **stream_kwargs)
            )
        except TypeError:
            # Some runtimes may not accept the 'subgraphs' kwarg. Retry without mutating
            # the original kwargs to keep behavior consistent with the sync path.
            stream_kwargs2 = dict(stream_kwargs)
            stream_kwargs2.pop("subgraphs", None)
            async_iterator = cast(
                AsyncIterable[Any], self.graph.astream(inputs, config or {}, **stream_kwargs2)
            )

        # Local per-run state, matching sync path
        last_values: Any | None = None
        last_decision_key: str | None = None
        updates_seen = 0
        single_mode_label: str | None = (
            self.stream_modes[0] if len(self.stream_modes) == 1 else None
        )
        agg_key: tuple[str, str | None, str | None] | None = None
        agg_chunks: list[dict[str, Any]] = []
        agg_text: list[str] = []
        agg_labels: dict[str, str] = {}
        agg_actor: str | None = None

        def flush_messages() -> None:
            nonlocal step, agg_key, agg_chunks, agg_text, agg_labels, agg_actor
            if agg_key is None:
                return
            chunks_payload: dict[str, Any] = {"chunks": agg_chunks}
            chunks_b = self._normalize_bytes(chunks_payload)
            chunks_ref = self.store.put_blob(self.run.run_id, step, BlobKind.STATE, chunks_b)
            payload: dict[str, Any] = {
                "message": {"content": "".join(agg_text)},
                "metadata": {"chunks_count": len(agg_chunks)},
                "chunks_ref": chunks_ref.model_dump(mode="json"),
            }
            payload_b2 = self._normalize_bytes(payload)
            out_blob2 = self.store.put_blob(self.run.run_id, step, BlobKind.OUTPUT, payload_b2)
            labels2 = dict(agg_labels)
            labels2["stream_mode"] = "messages"
            if agg_actor:
                labels2.setdefault("node", agg_actor)
            prompt_hash: str | None = None
            provider: str | None = None
            model: str | None = None
            params_meta: dict[str, Any] = {}
            tools_list: list[Any] = []
            ctx_messages: Any | None = None
            _prompt_hash_failed = False
            try:
                from ..codec import to_bytes as _to_bytes

                sources: list[Any] = []
                for ch in agg_chunks:
                    meta = ch.get("metadata") if isinstance(ch, dict) else None
                    if not isinstance(meta, dict):
                        continue
                    for key in ("llm_input_messages", "input_messages", "messages", "prompt"):
                        if key in meta:
                            sources.append(meta[key])
                            if ctx_messages is None:
                                ctx_messages = meta[key]
                    if provider is None and isinstance(meta.get("provider"), str):
                        provider = str(meta.get("provider"))
                    if model is None and isinstance(meta.get("model"), str):
                        model = str(meta.get("model"))
                    for p in ("temperature", "top_p", "tool_choice"):
                        val = meta.get(p)
                        if val is None and isinstance(meta.get("params"), dict):
                            params = meta["params"]
                            try:
                                val = params.get(p)
                            except Exception:
                                val = None
                        if val is not None and p not in params_meta:
                            if isinstance(val, str | int | float | bool):
                                params_meta[p] = val
                    # Collect tools list if exposed in metadata
                    try:
                        t = meta.get("tools") if isinstance(meta, dict) else None
                        if t is None and isinstance(meta, dict):
                            t = meta.get("available_tools")
                        if isinstance(t, list) and t:
                            tools_list.extend(t)
                    except Exception:
                        pass
                if sources:
                    prompt_hash = hash_bytes(_to_bytes({"sources": sources}))
            except Exception:
                prompt_hash = None
                _prompt_hash_failed = True
            try:
                anchor_id2 = self._make_anchor_id(ActionType.LLM, agg_actor or "graph", labels2)
                if anchor_id2:
                    labels2["anchor_id"] = anchor_id2
            except Exception:
                pass
            if prompt_hash:
                labels2["hash_source"] = "aggregated"
            # Compute optional tools digest + prompt context
            ev_hashes: dict[str, str] = {"output": out_blob2.sha256_hex}
            if prompt_hash:
                ev_hashes["prompt"] = prompt_hash
            ev_tools_digest: str | None = None
            input_ref = None
            try:
                if tools_list:
                    from ..codec import to_bytes as _to_bytes
                    from ..events import hash_bytes as _hash

                    ev_tools_digest = _hash(_to_bytes({"tools": tools_list}))
                    ev_hashes["tools"] = ev_tools_digest
                    if ctx_messages is not None:
                        ctx_obj = {"messages": ctx_messages, "tools": tools_list}
                        parts_b = self._normalize_bytes(ctx_obj)
                        input_ref = self.store.put_blob(
                            self.run.run_id, step, BlobKind.INPUT, parts_b
                        )
                        ev_hashes["prompt_ctx"] = _hash(_to_bytes(ctx_obj))
            except Exception:
                pass

            ev2 = Event(
                run_id=self.run.run_id,
                step=step,
                action_type=ActionType.LLM,
                actor=agg_actor or "graph",
                input_ref=input_ref,
                output_ref=out_blob2,
                hashes=ev_hashes,
                labels=labels2,
                model_meta={
                    "adapter_version": self.ADAPTER_VERSION,
                    "framework": "langgraph",
                    "chunks_count": len(agg_chunks),
                    "prompt_hash_agg_failed": True if _prompt_hash_failed else False,
                    **({"provider": provider} if provider else {}),
                    **({"model": model} if model else {}),
                    **params_meta,
                },
                ts=tw_now(),
            )
            if ev_tools_digest is not None:
                ev2 = ev2.model_copy(update={"tools_digest": ev_tools_digest})
            try:
                if "prompt" not in ev2.hashes:
                    staged = try_pop_prompt_hash()
                    if staged:
                        new_labels = dict(ev2.labels or {})
                        new_labels["hash_source"] = "staged"
                        ev2 = ev2.model_copy(
                            update={
                                "hashes": {**ev2.hashes, "prompt": staged},
                                "labels": new_labels,
                            }
                        )
            except Exception:
                pass
            self._append_event(ev2)
            step += 1
            agg_key = None
            agg_chunks = []
            agg_text = []
            agg_labels = {}
            agg_actor = None

        async for update in async_iterator:
            namespace_label, stream_mode_label, upd = self._normalize_stream_item(
                update, single_mode_label
            )

            # Identify actor
            actor = self._infer_actor(upd)
            if stream_mode_label == "messages" or single_mode_label == "messages":
                key: tuple[str, str | None, str | None] = (
                    actor or "graph",
                    namespace_label,
                    thread_id,
                )
                if agg_key is None:
                    agg_key = key
                elif agg_key != key:
                    flush_messages()
                    agg_key = key
                normalized: dict[str, Any]
                if isinstance(upd, tuple | list) and len(upd) == 2:
                    normalized = self._serialize_messages_tuple(upd)
                elif isinstance(upd, dict) and (
                    "message" in upd and "metadata" in upd and isinstance(upd["metadata"], dict)
                ):
                    normalized = upd  # already normalized
                else:
                    normalized = {"message": upd, "metadata": {}}
                # Propagate labels for aggregated messages event
                agg_labels = {}
                if thread_id:
                    agg_labels["thread_id"] = thread_id
                if namespace_label:
                    agg_labels["namespace"] = namespace_label
                # Infer actor label
                if actor and actor != "graph":
                    agg_actor = actor
                # Append chunk content
                agg_chunks.append(normalized)
                try:
                    msg = normalized.get("message")
                    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                        agg_text.append(msg["content"])
                    elif isinstance(msg, str):
                        agg_text.append(msg)
                except Exception:
                    pass
                continue

            # Flush pending messages if non-messages update arrives
            flush_messages()
            payload_b = self._normalize_bytes(upd)
            out_blob = self.store.put_blob(self.run.run_id, step, BlobKind.OUTPUT, payload_b)
            ev_labels: dict[str, str] = {}
            if stream_mode_label:
                ev_labels["stream_mode"] = stream_mode_label
            if namespace_label:
                ev_labels["namespace"] = namespace_label
            if thread_id:
                ev_labels.setdefault("thread_id", thread_id)
            if actor and actor != "graph":
                ev_labels.setdefault("node", actor)

            atype = self._infer_action_type(upd)
            if stream_mode_label == "messages" or single_mode_label == "messages":
                atype = ActionType.LLM
            tool_meta: dict[str, str] | None = self._classify_tool_from_update(upd)
            if tool_meta:
                atype = ActionType.TOOL

            base_meta = {"adapter_version": self.ADAPTER_VERSION, "framework": "langgraph"}
            ev_meta: dict[str, Any] | None = None

            ev_hashes: dict[str, str] = {"output": out_blob.sha256_hex}
            input_ref = None
            ev_tools_digest: str | None = None
            if atype is ActionType.LLM:
                try:
                    from ..codec import to_bytes as _to_bytes

                    cand = None
                    if isinstance(upd, dict):
                        for pkey in ("llm_input_messages", "input_messages", "messages", "prompt"):
                            if pkey in upd:
                                cand = upd[pkey]
                                break
                    if cand is not None:
                        ev_hashes["prompt"] = hash_bytes(_to_bytes(cand))
                        try:
                            ev_labels["hash_source"] = "aggregated"
                        except Exception:
                            pass
                except Exception:
                    pass
                try:
                    if "prompt" not in ev_hashes:
                        staged_p = try_pop_prompt_hash()
                        if staged_p:
                            ev_hashes["prompt"] = staged_p
                            try:
                                ev_labels["hash_source"] = "staged"
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    meta_obj = upd.get("metadata") if isinstance(upd, dict) else None
                    if isinstance(meta_obj, dict):
                        try:
                            prov = meta_obj.get("provider")
                            mdl = meta_obj.get("model")
                            if prov is not None:
                                ev_meta = ev_meta or {}
                                if isinstance(prov, str):
                                    ev_meta["provider"] = prov
                            if mdl is not None:
                                ev_meta = ev_meta or {}
                                if isinstance(mdl, str):
                                    ev_meta["model"] = mdl
                        except Exception:
                            pass
                        for p in ("temperature", "top_p", "tool_choice"):
                            val = meta_obj.get(p)
                            if val is None and isinstance(meta_obj.get("params"), dict):
                                params2 = meta_obj["params"]
                                try:
                                    val = params2.get(p)
                                except Exception:
                                    val = None
                            if val is not None:
                                ev_meta = ev_meta or {}
                                if isinstance(val, str | int | float | bool):
                                    ev_meta[p] = val
                except Exception:
                    pass
            if atype is ActionType.TOOL:
                try:
                    from ..codec import to_bytes as _to_bytes

                    args_env = self._extract_tool_args(upd)
                    if args_env is not None:
                        ev_hashes["args"] = hash_bytes(_to_bytes(args_env))
                except Exception:
                    pass
                try:
                    if "args" not in ev_hashes:
                        staged_a = try_pop_tool_args_hash()
                        if staged_a:
                            ev_hashes["args"] = staged_a
                except Exception:
                    pass

            try:
                tool_nm = None
                if isinstance(upd, dict):
                    tool_nm = str(upd.get("tool_name")) if upd.get("tool_name") else None
                anchor_id = self._make_anchor_id(atype, actor, ev_labels, tool_nm)
                if anchor_id:
                    ev_labels["anchor_id"] = anchor_id
            except Exception:
                pass

            ev = Event(
                run_id=self.run.run_id,
                step=step,
                action_type=atype,
                actor=actor,
                input_ref=(
                    input_ref if atype is ActionType.LLM and input_ref is not None else None
                ),
                output_ref=out_blob,
                hashes=ev_hashes,
                labels=ev_labels,
                model_meta=(base_meta if ev_meta is None else {**base_meta, **ev_meta}),
                ts=tw_now(),
            )
            if tool_meta:
                ev = ev.model_copy(update=tool_meta)
            if atype is ActionType.LLM and ev_tools_digest is not None:
                ev = ev.model_copy(update={"tools_digest": ev_tools_digest})

            self._append_event(ev)
            step += 1
            updates_seen += 1

            # MEMORY synthesis from values stream when configured (async)
            try:
                if (stream_mode_label == "values" or single_mode_label == "values") and isinstance(
                    upd, dict
                ):
                    vals = self._extract_values(upd)
                    if isinstance(vals, dict):
                        if self.memory_keys:
                            step = self._emit_memory_events(
                                step=step,
                                actor=actor,
                                namespace_label=namespace_label,
                                thread_id=thread_id,
                                values=vals,
                            )
                        if self.detect_retrieval:
                            env = self._detect_retrieval(vals)
                            if isinstance(env, dict):
                                step = self._emit_retrieval_event(
                                    step=step,
                                    actor=actor,
                                    namespace_label=namespace_label,
                                    thread_id=thread_id,
                                    env=env,
                                )
            except Exception:
                pass

            try:
                if (stream_mode_label == "values" or single_mode_label == "values") and isinstance(
                    upd, dict
                ):
                    next_nodes = self._extract_next_nodes(upd)
                    if next_nodes is not None:
                        decision_key = "|".join(next_nodes)
                        if decision_key != last_decision_key:
                            dec_labels: dict[str, str] = {}
                            if namespace_label:
                                dec_labels["namespace"] = namespace_label
                            if thread_id:
                                dec_labels["thread_id"] = thread_id
                            if actor and actor != "graph":
                                dec_labels["node"] = actor
                            try:
                                dec_anchor = self._make_anchor_id(
                                    ActionType.DECISION, actor, dec_labels
                                )
                                if dec_anchor:
                                    dec_labels["anchor_id"] = dec_anchor
                            except Exception:
                                pass
                            dec_labels["decision"] = decision_key
                            dec_ev = Event(
                                run_id=self.run.run_id,
                                step=step,
                                action_type=ActionType.DECISION,
                                actor=actor or "router",
                                labels=dec_labels,
                                model_meta={
                                    "adapter_version": self.ADAPTER_VERSION,
                                    "framework": "langgraph",
                                },
                                ts=tw_now(),
                            )
                            self._append_event(dec_ev)
                            step += 1
                            last_decision_key = decision_key
                            try:
                                if "decision" in self.snapshot_on:
                                    snap_payload_dec: Any | None = None
                                    if last_values is not None and isinstance(
                                        last_values, dict | list
                                    ):
                                        snap_payload_dec = last_values
                                    else:
                                        get_state = getattr(self.graph, "get_state", None)
                                        if callable(get_state) and config:
                                            snapshot = get_state(config)
                                            snap_payload_dec = self._extract_values(snapshot)
                                    if snap_payload_dec is not None:
                                        extra_labels_dec: dict[str, str] = {}
                                        if thread_id:
                                            extra_labels_dec["thread_id"] = thread_id
                                        cpd = self._extract_checkpoint_id(config, snap_payload_dec)
                                        if cpd is not None:
                                            extra_labels_dec["checkpoint_id"] = cpd
                                        self._persist_snapshot(
                                            step, snap_payload_dec, labels_extra=extra_labels_dec
                                        )
                                        step += 1
                            except Exception:
                                pass
            except Exception:
                pass
            try:
                if (stream_mode_label == "values" or single_mode_label == "values") and isinstance(
                    upd, dict
                ):
                    last_values = upd
            except Exception:
                pass

            if self.snapshot_every > 0 and updates_seen % self.snapshot_every == 0:
                snap_payload: Any | None = None
                if last_values is not None and isinstance(last_values, dict | list):
                    snap_payload = last_values
                else:
                    try:
                        get_state = getattr(self.graph, "get_state", None)
                        if callable(get_state) and config:
                            snapshot = get_state(config)
                            snap_payload = self._extract_values(snapshot)
                    except Exception:
                        snap_payload = None
                if snap_payload is not None:
                    extra_labels: dict[str, str] = {}
                    if thread_id:
                        extra_labels["thread_id"] = thread_id
                    cp = self._extract_checkpoint_id(config, snap_payload)
                    if cp is not None:
                        extra_labels["checkpoint_id"] = cp
                    self._persist_snapshot(step, snap_payload, labels_extra=extra_labels)
                    step += 1

            try:
                if isinstance(upd, dict) and "__interrupt__" in upd:
                    _hitl_payload = {"hitl": {"type": "interrupt", "payload": upd["__interrupt__"]}}
                    hitl_blob = self.store.put_blob(
                        self.run.run_id,
                        step,
                        BlobKind.OUTPUT,
                        self._normalize_bytes(_hitl_payload),
                    )
                    hitl_ev = Event(
                        run_id=self.run.run_id,
                        step=step,
                        action_type=ActionType.HITL,
                        actor=actor,
                        output_ref=hitl_blob,
                        hashes={"output": hitl_blob.sha256_hex},
                        labels=(ev_labels if ev_labels else {}),
                        model_meta={
                            "adapter_version": self.ADAPTER_VERSION,
                            "framework": "langgraph",
                        },
                        ts=tw_now(),
                    )
                    self._append_event(hitl_ev)
                    step += 1
            except Exception:
                pass

        # Flush remaining aggregated messages
        flush_messages()

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
        except Exception:
            terminal_state = None

        if terminal_state is not None and "terminal" in (self.snapshot_on or {"terminal"}):
            extra_labels2: dict[str, str] = {}
            if thread_id:
                extra_labels2["thread_id"] = thread_id
            cp2 = self._extract_checkpoint_id(config, terminal_state)
            if cp2 is not None:
                extra_labels2["checkpoint_id"] = cp2
            self._persist_snapshot(step, terminal_state, labels_extra=extra_labels2)

        # Ensure any pending batched events are flushed; offload to thread to avoid blocking loop
        await asyncio.to_thread(self._flush_events)

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
            model_meta={"adapter_version": self.ADAPTER_VERSION, "framework": "langgraph"},
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
            model_meta={"adapter_version": self.ADAPTER_VERSION, "framework": "langgraph"},
            ts=tw_now(),
        )
        self._append_event(ev)
        return labels, thread_id, step + 1

    def _build_stream_kwargs(
        self, config: dict[str, Any] | None, thread_id: str | None
    ) -> dict[str, Any]:
        effective_durability = self.durability
        if effective_durability is None and thread_id:
            effective_durability = "sync"
        stream_kwargs: dict[str, Any] = {
            "stream_mode": list(self.stream_modes)
            if len(self.stream_modes) > 1
            else (self.stream_modes[0] if self.stream_modes else "updates"),
        }
        if self.stream_subgraphs:
            stream_kwargs["subgraphs"] = True
        if effective_durability is not None:
            stream_kwargs["durability"] = effective_durability
        return stream_kwargs

    def _iter_stream_sync(
        self,
        graph: Any,
        inputs: dict[str, Any],
        config: dict[str, Any] | None,
        stream_kwargs: dict[str, Any],
    ) -> Iterable[Any]:
        from typing import cast

        try:
            it = graph.stream(inputs, config or {}, **stream_kwargs)
            return cast(Iterable[Any], it)
        except TypeError:
            stream_kwargs2 = dict(stream_kwargs)
            stream_kwargs2.pop("subgraphs", None)
            it2 = graph.stream(inputs, config or {}, **stream_kwargs2)
            return cast(Iterable[Any], it2)

    # Event batching helpers
    _pending_events: list[Event] = field(default_factory=list, init=False, repr=False)
    # Memory synthesis state (path -> last item hash)
    _tw_mem_prev: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def _append_event(self, ev: Event) -> None:
        # Append with optional batching
        bs = max(1, int(self.event_batch_size))
        if bs == 1:
            self.store.append_event(ev)
            return
        pending = self._pending_events
        pending.append(ev)
        if len(pending) >= bs:
            self.store.append_events(pending)
            pending.clear()

    def _flush_events(self) -> None:
        pending = self._pending_events
        if pending:
            self.store.append_events(pending)
            pending.clear()

    def _normalize_bytes(self, obj: Any) -> bytes:
        from ..codec import to_bytes
        from ..events import redact

        # Apply redaction when possible (best effort: no privacy marks out of band here)
        redacted = obj
        try:
            redacted = redact(obj, self.privacy_marks)
        except Exception:
            pass
        try:
            return to_bytes(redacted)
        except Exception:
            # Fallbacks for non-JSON-serializable objects
            # - StateSnapshot-like: use .values
            vals = self._extract_values(redacted)
            if vals is not None:
                return to_bytes(vals)
            # - Pydantic v2 models
            try:
                if hasattr(redacted, "model_dump"):
                    md = redacted.model_dump
                    if callable(md):
                        return to_bytes(md(mode="json"))
            except Exception:
                pass
            # - last resort: stable-ish repr envelope
            return to_bytes({"_repr": repr(redacted)})

    # --- memory/tools helpers ---
    def _get_by_path(self, root: dict[str, Any], path: str) -> Any | None:
        try:
            cur: Any = root
            for seg in path.split("."):
                if not isinstance(cur, dict) or seg not in cur:
                    return None
                cur = cur[seg]
            return cur
        except Exception:
            return None

    def _prune_mem_value(self, value: Any) -> Any:
        if self.memory_pruner is None:
            return value
        try:
            pruned = self.memory_pruner(value)
            return pruned if isinstance(pruned, dict | list | str | int | float | bool) else value
        except Exception:
            return value

    def _infer_mem_scope_from_path(self, path: str) -> str:
        p = path.lower()
        if "long" in p:
            return "long"
        if "short" in p:
            return "short"
        return "working"

    def _emit_memory_events(
        self,
        *,
        step: int,
        actor: str,
        namespace_label: str | None,
        thread_id: str | None,
        values: dict[str, Any],
    ) -> int:
        # Maintain last-seen memory item hashes on the instance
        mem_prev: dict[str, str]
        if not hasattr(self, "_tw_mem_prev"):
            self._tw_mem_prev = {}
        mem_prev = self._tw_mem_prev
        for path in self.memory_keys:
            v = self._get_by_path(values, path)
            if v is None:
                continue
            pruned = self._prune_mem_value(v)
            payload = {"key": path, "value": pruned}
            data_b = self._normalize_bytes(payload)
            blob = self.store.put_blob(self.run.run_id, step, BlobKind.MEMORY, data_b)
            h = blob.sha256_hex
            prev = mem_prev.get(path)
            if prev is not None and prev == h:
                continue  # unchanged
            mem_prev[path] = h
            labels: dict[str, str] = {}
            if namespace_label:
                labels["namespace"] = namespace_label
            if thread_id:
                labels["thread_id"] = thread_id
            if actor and actor != "graph":
                labels["node"] = actor
            labels["mem_op"] = "PUT" if prev is None else "UPDATE"
            labels["mem_scope"] = self._infer_mem_scope_from_path(path)
            labels["mem_space"] = actor or "graph"
            try:
                labels["anchor_id"] = self._make_anchor_id(ActionType.MEMORY, actor, labels)
            except Exception:
                pass
            ev = Event(
                run_id=self.run.run_id,
                step=step,
                action_type=ActionType.MEMORY,
                actor=actor or "graph",
                output_ref=blob,
                hashes={"item": h},
                labels=labels,
                model_meta={
                    "adapter_version": self.ADAPTER_VERSION,
                    "framework": "langgraph",
                    "mem_provider": "LangGraphState",
                },
                ts=tw_now(),
            )
            # top-level convenience
            ev = ev.model_copy(update={"mem_provider": "LangGraphState"})
            self._append_event(ev)
            step += 1
        return step

    # --- retrieval detection & emission ---
    def _detect_retrieval(self, values_like: Any) -> dict[str, Any] | None:
        """Best-effort detection of retrieval query/results within a values/update payload.

        Returns a normalized envelope: {"query", "items", "top_k", "retriever", "query_id"}
        or None when not detected.
        """
        try:
            src = values_like
            if isinstance(src, dict) and "values" in src and isinstance(src["values"], dict):
                src = src["values"]
            if not isinstance(src, dict):
                return None
            # Direct envelope
            env: dict[str, Any] | None = None
            cand = src.get("retrieval")
            if isinstance(cand, dict):
                env = dict(cand)
            # Generic shapes with query + results/documents/docs
            if env is None:
                if "query" in src and isinstance(src.get("results"), list):
                    env = {"query": src.get("query"), "items": src.get("results")}
                elif "query" in src and isinstance(src.get("documents"), list):
                    env = {"query": src.get("query"), "items": src.get("documents")}
                elif "query" in src and isinstance(src.get("docs"), list):
                    env = {"query": src.get("query"), "items": src.get("docs")}
            if env is None:
                return None

            # Normalize items list -> list of {id, text, score, metadata}
            items_raw = env.get("items") or env.get("results")
            if not isinstance(items_raw, list) or not items_raw:
                return None

            def _norm_item(idx: int, it: Any) -> dict[str, Any]:
                try:
                    if not isinstance(it, dict):
                        return {"id": str(idx), "text": str(it)}
                    meta = it.get("metadata") or it.get("meta") or {}
                    # text candidates
                    txt = (
                        it.get("text")
                        or it.get("content")
                        or it.get("page_content")
                        or it.get("document")
                        or (meta.get("text") if isinstance(meta, dict) else None)
                    )
                    # id candidates
                    iid = it.get("id") or (meta.get("id") if isinstance(meta, dict) else None)
                    # score from common fields
                    score = it.get("score")
                    if score is None:
                        # sometimes similarity/distance used
                        s = it.get("similarity")
                        d = it.get("distance")
                        score = (
                            s
                            if isinstance(s, int | float)
                            else (d if isinstance(d, int | float) else None)
                        )
                    out = {
                        "id": str(iid) if iid is not None else str(idx),
                        "text": str(txt) if txt is not None else "",
                        "metadata": meta if isinstance(meta, dict) else {},
                    }
                    if isinstance(score, int | float):
                        out["score"] = float(score)
                    return out
                except Exception:
                    return {"id": str(idx), "text": ""}

            norm_items = [_norm_item(i, it) for i, it in enumerate(items_raw)]
            query = env.get("query")
            retriever = env.get("retriever") or src.get("retriever")
            top_k = env.get("top_k") or src.get("top_k")
            qid = env.get("query_id") or src.get("query_id")
            out_env: dict[str, Any] = {
                "query": query,
                "items": norm_items,
                "top_k": int(top_k)
                if isinstance(top_k, int)
                else (int(top_k) if isinstance(top_k, str) and top_k.isdigit() else None),
                "retriever": str(retriever) if isinstance(retriever, str | int) else None,
                "query_id": str(qid) if isinstance(qid, str | int) else None,
            }

            # Apply optional custom detector transformation or pruner
            if self.retrieval_detector is not None:
                try:
                    custom = self.retrieval_detector(src)
                    if isinstance(custom, dict) and custom.get("items"):
                        out_env = custom
                except Exception:
                    pass
            if self.retrieval_pruner is not None:
                try:
                    pruned = self.retrieval_pruner(out_env)
                    if isinstance(pruned, dict) and pruned.get("items"):
                        out_env = pruned
                    elif isinstance(pruned, list):
                        out_env = dict(out_env)
                        out_env["items"] = pruned
                except Exception:
                    pass
            return out_env
        except Exception:  # pragma: no cover
            return None

    def _emit_retrieval_event(
        self,
        *,
        step: int,
        actor: str,
        namespace_label: str | None,
        thread_id: str | None,
        env: dict[str, Any],
    ) -> int:
        try:
            items = env.get("items")
            if not isinstance(items, list) or not items:
                return step
            query = env.get("query")
            retriever = env.get("retriever")
            top_k = env.get("top_k")
            query_id = env.get("query_id")
            payload = {
                "query": query,
                "items": items,
                "policy": {"retriever": retriever, "top_k": top_k},
            }
            data_b = self._normalize_bytes(payload)
            blob = self.store.put_blob(self.run.run_id, step, BlobKind.MEMORY, data_b)
            from ..codec import to_bytes as _to_bytes

            hashes = {}
            try:
                if query is not None:
                    hashes["query"] = hash_bytes(_to_bytes(query))
            except Exception:
                pass
            try:
                hashes["results"] = hash_bytes(_to_bytes({"items": items}))
            except Exception:
                pass
            labels: dict[str, str] = {}
            if namespace_label:
                labels["namespace"] = namespace_label
            if thread_id:
                labels["thread_id"] = thread_id
            if actor and actor != "graph":
                labels["node"] = actor
            try:
                labels["anchor_id"] = self._make_anchor_id(ActionType.RETRIEVAL, actor, labels)
            except Exception:
                pass
            ev = Event(
                run_id=self.run.run_id,
                step=step,
                action_type=ActionType.RETRIEVAL,
                actor=actor or "graph",
                output_ref=blob,
                hashes=hashes,
                labels=labels,
                model_meta={
                    "adapter_version": self.ADAPTER_VERSION,
                    "framework": "langgraph",
                    "mem_provider": "LangGraphState",
                },
                ts=tw_now(),
            )
            # Convenience top-level fields for filtering
            ev = ev.model_copy(
                update={
                    "retriever": retriever if isinstance(retriever, str) else None,
                    "top_k": int(top_k) if isinstance(top_k, int) else None,
                    "query_id": str(query_id) if isinstance(query_id, str | int) else None,
                }
            )
            self._append_event(ev)
            return step + 1
        except Exception:  # pragma: no cover
            return step

    def _extract_tools_from_update(self, update: Any) -> list[Any] | None:
        try:
            obs = update
            if isinstance(update, dict) and len(update) == 1:
                try:
                    ((_, inner),) = update.items()
                    if isinstance(inner, dict):
                        obs = inner
                except Exception:
                    obs = update
            if not isinstance(obs, dict):
                return None
            meta = obs.get("metadata") if isinstance(obs.get("metadata"), dict) else None
            tools = obs.get("tools")
            if tools is None and isinstance(meta, dict):
                tools = meta.get("tools") or meta.get("available_tools")
            if isinstance(tools, list) and tools:
                return tools
        except Exception:
            return None
        return None

    def _serialize_messages_tuple(self, pair: Any) -> dict[str, Any]:
        """Normalize a (message_chunk, metadata) pair into JSON-serializable dict.

        Best-effort extraction of text content and relevant metadata fields.
        """
        msg = pair[0]
        meta = pair[1]

        def to_plain(x: Any) -> Any:
            try:
                # Pydantic v2 model_dump support
                if hasattr(x, "model_dump") and callable(x.model_dump):
                    return x.model_dump(mode="json")
            except Exception:
                pass
            try:
                # LangChain AIMessageChunk has .content
                if hasattr(x, "content"):
                    return {"content": x.content}
            except Exception:
                pass
            if isinstance(x, str | int | float | bool):
                return x
            if isinstance(x, dict):
                return x
            if isinstance(x, list | tuple):
                return [to_plain(y) for y in x]
            return {"_repr": repr(x)}

        return {"message": to_plain(msg), "metadata": to_plain(meta)}

    def _normalize_stream_item(
        self, update: Any, single_mode_label: str | None
    ) -> tuple[str | None, str | None, Any]:
        """Normalize a stream update into (namespace_label, stream_mode_label, data).

        Supports shapes:
        - (namespace, mode, data)
        - (mode, data)
        - (namespace, data)
        - data-only (fallback to provided single_mode_label)
        """
        namespace_label: str | None = None
        stream_mode_label: str | None = None
        upd = update
        try:
            if (
                isinstance(update, tuple | list)
                and len(update) == 3
                and isinstance(update[0], tuple | list)
                and isinstance(update[1], str)
            ):
                ns = [str(x) for x in update[0]]
                namespace_label = "/".join(ns)
                stream_mode_label = update[1]
                upd = update[2]
                return namespace_label, stream_mode_label, upd
        except Exception:
            pass
        try:
            if isinstance(update, tuple | list) and len(update) == 2 and isinstance(update[0], str):
                stream_mode_label = update[0]
                upd = update[1]
                return namespace_label, stream_mode_label, upd
        except Exception:
            pass
        try:
            if (
                isinstance(update, tuple | list)
                and len(update) == 2
                and isinstance(update[0], tuple | list)
            ):
                ns = [str(x) for x in update[0]]
                namespace_label = "/".join(ns)
                upd = update[1]
                return namespace_label, stream_mode_label, upd
        except Exception:
            pass
        # Fallback: best guess using single mode label
        return namespace_label, single_mode_label, upd

    def _derive_actor_from_namespace(self, namespace_label: str, actor: str) -> str:
        """Derive a node actor name from a namespace path.

        Namespace segments are '/' separated. The final segment may include a subgraph
        instance suffix after ':'.
        """
        try:
            last_seg = namespace_label.split("/")[-1]
            return last_seg.split(":")[0] if ":" in last_seg else last_seg
        except Exception:  # pragma: no cover - defensive
            return actor

    def _finalize_messages_aggregate(
        self,
        *,
        step: int,
        agg_actor: str | None,
        agg_text: list[str],
        agg_chunks: list[dict[str, Any]],
        agg_labels: dict[str, str],
    ) -> tuple[Event, int]:
        """Create aggregated LLM Event for messages buffer and return (event, next_step)."""
        # Persist chunks separately to keep main payload compact
        chunks_payload: dict[str, Any] = {"chunks": agg_chunks}
        chunks_b = self._normalize_bytes(chunks_payload)
        chunks_ref = self.store.put_blob(self.run.run_id, step, BlobKind.STATE, chunks_b)
        payload: dict[str, Any] = {
            "message": {"content": "".join(agg_text)},
            "metadata": {"chunks_count": len(agg_chunks)},
            "chunks_ref": chunks_ref.model_dump(mode="json"),
        }
        payload_b2 = self._normalize_bytes(payload)
        out_blob2 = self.store.put_blob(self.run.run_id, step, BlobKind.OUTPUT, payload_b2)
        labels2 = dict(agg_labels)
        labels2["stream_mode"] = "messages"
        if agg_actor:
            labels2.setdefault("node", agg_actor)

        # Best-effort extraction for prompt hash, provider/model/params, tools
        prompt_hash: str | None = None
        provider: str | None = None
        model: str | None = None
        params_meta: dict[str, Any] = {}
        tools_list: list[Any] = []
        ctx_messages: Any | None = None
        _prompt_hash_failed = False
        try:
            from ..codec import to_bytes as _to_bytes

            sources: list[Any] = []
            for ch in agg_chunks:
                meta = ch.get("metadata") if isinstance(ch, dict) else None
                if not isinstance(meta, dict):
                    continue
                for key in ("llm_input_messages", "input_messages", "messages", "prompt"):
                    if key in meta:
                        sources.append(meta[key])
                        if ctx_messages is None:
                            ctx_messages = meta[key]
                if provider is None and isinstance(meta.get("provider"), str):
                    provider = str(meta.get("provider"))
                if model is None and isinstance(meta.get("model"), str):
                    model = str(meta.get("model"))
                for p in ("temperature", "top_p", "tool_choice"):
                    val = meta.get(p)
                    if val is None and isinstance(meta.get("params"), dict):
                        params = meta["params"]
                        try:
                            val = params.get(p)
                        except Exception:
                            val = None
                    if val is not None and p not in params_meta:
                        if isinstance(val, str | int | float | bool):
                            params_meta[p] = val
                # Tools: prefer explicit list under tools/available_tools
                try:
                    t = meta.get("tools") if isinstance(meta, dict) else None
                    if t is None and isinstance(meta, dict):
                        t = meta.get("available_tools")
                    if isinstance(t, list) and t:
                        tools_list.extend(t)
                except Exception:
                    pass
            if sources:
                prompt_hash = hash_bytes(_to_bytes({"sources": sources}))
        except Exception:
            prompt_hash = None
            _prompt_hash_failed = True

        # Anchor id
        try:
            anchor_id2 = self._make_anchor_id(ActionType.LLM, agg_actor or "graph", labels2)
            if anchor_id2:
                labels2["anchor_id"] = anchor_id2
        except Exception:
            pass

        if prompt_hash:
            labels2["hash_source"] = "aggregated"

        # Compute optional tools digest + prompt context blob
        ev_hashes: dict[str, str] = {"output": out_blob2.sha256_hex}
        if prompt_hash:
            ev_hashes["prompt"] = prompt_hash
        ev_tools_digest: str | None = None
        input_ref = None
        try:
            if tools_list:
                from ..codec import to_bytes as _to_bytes
                from ..events import hash_bytes as _hash

                # Normalize as a stable envelope for hashing
                ev_tools_digest = _hash(_to_bytes({"tools": tools_list}))
                ev_hashes["tools"] = ev_tools_digest
                if ctx_messages is not None:
                    ctx_obj = {"messages": ctx_messages, "tools": tools_list}
                    parts_b = self._normalize_bytes(ctx_obj)
                    input_ref = self.store.put_blob(self.run.run_id, step, BlobKind.INPUT, parts_b)
                    ev_hashes["prompt_ctx"] = _hash(_to_bytes(ctx_obj))
        except Exception:
            pass

        ev2 = Event(
            run_id=self.run.run_id,
            step=step,
            action_type=ActionType.LLM,
            actor=agg_actor or "graph",
            input_ref=input_ref,
            output_ref=out_blob2,
            hashes=ev_hashes,
            labels=labels2,
            model_meta={
                "adapter_version": self.ADAPTER_VERSION,
                "framework": "langgraph",
                "chunks_count": len(agg_chunks),
                "prompt_hash_agg_failed": True if _prompt_hash_failed else False,
                **({"provider": provider} if provider else {}),
                **({"model": model} if model else {}),
                **params_meta,
            },
            ts=tw_now(),
        )
        if ev_tools_digest is not None:
            ev2 = ev2.model_copy(update={"tools_digest": ev_tools_digest})
        # Merge a staged prompt hash when not present
        try:
            if "prompt" not in ev2.hashes:
                staged = try_pop_prompt_hash()
                if staged:
                    new_labels = dict(ev2.labels or {})
                    new_labels["hash_source"] = "staged"
                    ev2 = ev2.model_copy(
                        update={
                            "hashes": {**ev2.hashes, "prompt": staged},
                            "labels": new_labels,
                        }
                    )
        except Exception:
            pass
        return ev2, step + 1

    def _infer_actor(self, update: Any) -> str:
        # Best-effort extraction of node name from LangGraph update dicts
        if isinstance(update, dict):
            # Common "updates" shape: {"node_name": {...}}
            try:
                keys = list(update.keys())
                if len(keys) == 1 and isinstance(keys[0], str):
                    return keys[0]
            except Exception:
                pass
            node = update.get("node") or update.get("ns")
            if isinstance(node, str):
                return node
            if isinstance(node, list) and node:
                return str(node[-1])
        return "graph"

    def _infer_action_type(self, update: Any) -> ActionType:
        # Heuristic classification: messages stream or token-like -> LLM; tool -> TOOL; else SYS
        try:
            obs = update
            if isinstance(update, dict) and len(update) == 1:
                try:
                    ((_, inner),) = update.items()
                    if isinstance(inner, dict):
                        obs = inner
                except Exception:
                    obs = update
            if isinstance(obs, dict):
                if "tool" in obs or "tool_name" in obs or obs.get("tool_kind"):
                    return ActionType.TOOL
                # messages stream commonly includes token tuples or message metadata
                if "messages" in obs or "llm_input_messages" in obs:
                    return ActionType.LLM
        except Exception:
            pass
        return ActionType.SYS

    def _classify_tool_from_update(self, update: Any) -> dict[str, str] | None:
        # Allow user-provided classifier to recognize MCP calls (recommended)
        obs = update
        try:
            if isinstance(update, dict) and len(update) == 1:
                ((_, inner),) = update.items()
                if isinstance(inner, dict):
                    obs = inner
        except Exception:
            obs = update
        if self.tool_classifier and hasattr(obs, "get"):
            # If the update contains a reference to the tool object
            tool_obj = obs.get("tool") if isinstance(obs, dict) else None
            if tool_obj is not None:
                meta = self.tool_classifier(tool_obj)
                if meta:
                    # Normalize keys expected by Event and drop None values
                    out: dict[str, str] = {
                        "tool_kind": meta.get("tool_kind", "MCP") or "MCP",
                        "tool_name": meta.get("tool_name", "unknown") or "unknown",
                    }
                    if meta.get("mcp_server") is not None:
                        out["mcp_server"] = str(meta["mcp_server"])  # ensure str
                    if meta.get("mcp_transport") is not None:
                        out["mcp_transport"] = str(meta["mcp_transport"])  # ensure str
                    return out
        # Heuristic for messages-mode structures
        if isinstance(obs, dict):
            name = obs.get("tool_name") or obs.get("name")
            kind = obs.get("tool_kind")
            if name and (kind == "MCP" or update.get("mcp_server") or update.get("mcp_transport")):
                out2: dict[str, str] = {
                    "tool_kind": str(kind or "MCP"),
                    "tool_name": str(name),
                }
                if obs.get("mcp_server"):
                    out2["mcp_server"] = str(obs.get("mcp_server"))
                if obs.get("mcp_transport"):
                    out2["mcp_transport"] = str(obs.get("mcp_transport"))
                return out2
            # Additionally inspect nested metadata produced by messages stream
            meta = obs.get("metadata")
            if isinstance(meta, dict):
                name2 = meta.get("tool_name") or meta.get("name")
                kind2 = meta.get("tool_kind")
                if name2 and (
                    kind2 == "MCP" or meta.get("mcp_server") or meta.get("mcp_transport")
                ):
                    out3: dict[str, str] = {
                        "tool_kind": str(kind2 or "MCP"),
                        "tool_name": str(name2),
                    }
                    if meta.get("mcp_server"):
                        out3["mcp_server"] = str(meta.get("mcp_server"))
                    if meta.get("mcp_transport"):
                        out3["mcp_transport"] = str(meta.get("mcp_transport"))
                    return out3
        return None

    def _extract_values(self, snapshot_or_obj: Any) -> dict[str, Any] | None:
        """Best-effort extraction of state values from a LangGraph StateSnapshot-like object."""
        try:
            # StateSnapshot(values=..., ...)
            if hasattr(snapshot_or_obj, "values"):
                vals = snapshot_or_obj.values
                if isinstance(vals, dict):
                    return vals
            # Dict form
            if isinstance(snapshot_or_obj, dict):
                v = snapshot_or_obj.get("values")
                if isinstance(v, dict):
                    vals2: dict[str, Any] = v
                    return vals2
        except Exception:
            return None
        return None

    def _extract_next_nodes(self, values_like: Any) -> list[str] | None:
        """Try to pull a LangGraph-style `next` list from a values update payload.

        Accepts shapes like {"next": [...] } or {"values": {"next": [...]}}.
        """
        try:
            src = values_like
            if isinstance(src, dict) and "values" in src and isinstance(src["values"], dict):
                src = src["values"]
            if isinstance(src, dict):
                nxt = src.get("next")
                if isinstance(nxt, list):
                    out: list[str] = []
                    for x in nxt:
                        if isinstance(x, str):
                            out.append(x)
                        else:
                            try:
                                out.append(str(x))
                            except Exception:
                                return None
                    return out
        except Exception:
            return None
        return None

    def _extract_checkpoint_id(
        self, config: dict[str, Any] | None, state_or_values: Any
    ) -> str | None:
        """Best-effort extraction of checkpoint_id from values payload or graph.get_state(config).

        Looks for config.configurable.checkpoint_id within provided state/values or config.
        """
        # Try from provided state/values
        try:
            src = state_or_values
            if isinstance(src, dict):
                cfg = src.get("config")
                if isinstance(cfg, dict):
                    conf2 = cfg.get("configurable")
                    if isinstance(conf2, dict) and isinstance(
                        conf2.get("checkpoint_id"), str | int
                    ):
                        return str(conf2.get("checkpoint_id"))
        except Exception:
            pass
        # Try from passed config mapping
        try:
            if isinstance(config, dict):
                conf = config.get("configurable")
                if isinstance(conf, dict) and isinstance(conf.get("checkpoint_id"), str | int):
                    return str(conf.get("checkpoint_id"))
        except Exception:
            pass
        return None

    def _make_anchor_id(
        self,
        action_type: ActionType,
        actor: str,
        labels: dict[str, str],
        tool_name: str | None = None,
    ) -> str:
        tid = labels.get("thread_id", "")
        node = labels.get("node", actor or "")
        ns = labels.get("namespace", "")
        tool = f":{tool_name}" if tool_name else ""
        return f"{tid}:{node}:{ns}:{action_type.value}{tool}"

    def _extract_tool_args(self, update: Any) -> dict[str, Any] | None:
        """Best-effort extraction of tool call arguments for hashing.

        Normalizes into the envelope expected by PlaybackTool: {"args": [...], "kwargs": {...}}
        """
        try:
            obs = update
            if isinstance(update, dict) and len(update) == 1:
                try:
                    ((_, inner),) = update.items()
                    if isinstance(inner, dict):
                        obs = inner
                except Exception:
                    obs = update
            if not isinstance(obs, dict):
                return None
            if "args" in obs or "kwargs" in obs:
                args_v = obs.get("args", [])
                kwargs_v = obs.get("kwargs", {})
                # Ensure shapes are JSON-serializable
                if not isinstance(kwargs_v, dict):
                    try:
                        kwargs_v = {"_": kwargs_v}
                    except Exception:
                        kwargs_v = {}
                if isinstance(args_v, list):
                    norm_args = list(args_v)
                elif isinstance(args_v, tuple):
                    norm_args = list(args_v)
                else:
                    norm_args = [args_v]
                return {"args": norm_args, "kwargs": kwargs_v}
            # Alternative containers
            ta = obs.get("tool_args")
            if isinstance(ta, dict):
                a = ta.get("args", [])
                k = ta.get("kwargs", {})
                if not isinstance(k, dict):
                    k = {"_": k}
                if isinstance(a, list):
                    norm_a = list(a)
                elif isinstance(a, tuple):
                    norm_a = list(a)
                else:
                    norm_a = [a]
                return {"args": norm_a, "kwargs": k}
            inp = obs.get("input")
            if isinstance(inp, dict):
                return {"args": [], "kwargs": inp}
            params = obs.get("parameters")
            if isinstance(params, dict):
                return {"args": [], "kwargs": params}
        except Exception:
            return None
        return None


def _default_tool_classifier() -> ToolClassifier:
    """Return a best-effort classifier for MCP-like tools without hard deps.

    The returned callable inspects common attribute names on the tool object to
    identify MCP metadata. If none match, returns None to indicate unknown.
    """

    def _cls(tool: Any) -> dict[str, str] | None:
        try:
            # Name
            name = getattr(tool, "tool_name", None) or getattr(tool, "name", None)
            # Server hints
            server = (
                getattr(tool, "mcp_server", None)
                or getattr(tool, "server", None)
                or getattr(tool, "server_url", None)
            )
            # Transport hints
            transport = getattr(tool, "mcp_transport", None) or getattr(tool, "transport", None)
            # Some wrappers may tuck server into a sub-object with url/uri attribute
            try:
                if server is not None and not isinstance(server, str | int | float | bool):
                    # Attempt to pull url/uri
                    for k in ("url", "uri", "endpoint"):
                        v = getattr(server, k, None)
                        if isinstance(v, str):
                            server = v
                            break
            except Exception:
                pass
            if name or server or transport:
                out: dict[str, str] = {"tool_kind": "MCP"}
                if name:
                    out["tool_name"] = str(name)
                if server:
                    out["mcp_server"] = str(server)
                if transport:
                    out["mcp_transport"] = str(transport)
                return out
        except Exception:
            return None
        return None

    return _cls
