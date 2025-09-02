from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

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
    # Defaults per plan: capture updates + values (full state deltas); messages can be opted in
    stream_modes: Sequence[str] = ("updates", "values")
    stream_subgraphs: bool = True
    require_thread_id: bool = False
    durability: str | None = None  # e.g., "sync" | None; pass only when checkpointer present
    privacy_marks: dict[str, str] = field(default_factory=dict)

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

        # Register run
        self.store.create_run(self.run)
        step = 0

        # Capture initial RNG
        rng_before = snapshot_rng()

        # Persist an initial SYS event for input envelope
        input_blob = self.store.put_blob(
            self.run.run_id, step, BlobKind.INPUT, self._normalize_bytes(inputs)
        )
        # Try to capture thread_id (if present) for easier correlation
        thread_id: str | None = None
        if isinstance(config, dict):
            cfg = (
                config.get("configurable") if isinstance(config.get("configurable"), dict) else None
            )
            if isinstance(cfg, dict):
                tid = cfg.get("thread_id")
                thread_id = str(tid) if isinstance(tid, str | int) else None

        # Optional guard: when a checkpointer is expected, enforce thread_id presence
        if self.require_thread_id and not thread_id:
            raise ValueError(
                "require_thread_id=True but no configurable.thread_id provided in config"
            )

        # Build labels for recorder context
        labels: dict[str, str] = {}
        if thread_id:
            labels["thread_id"] = thread_id
        # For SYS/input, set a neutral node label for alignment
        labels["node"] = "graph"
        # Propagate branch lineage onto the initial SYS event when present on Run
        try:
            if isinstance(self.run.labels, dict) and "branch_of" in self.run.labels:
                bo = self.run.labels.get("branch_of")
                if isinstance(bo, str) and bo:
                    labels["branch_of"] = bo
        except Exception:
            pass
        # Record recorder configuration to aid filtering/debug
        try:
            # normalize stream modes label
            if self.stream_modes:
                sm = (
                    ",".join(self.stream_modes)
                    if len(self.stream_modes) > 1
                    else self.stream_modes[0]
                )
                labels["stream_mode"] = str(sm)
            if self.stream_subgraphs:
                labels["subgraphs"] = "true"
            # Determine effective durability: if thread_id present and not
            # explicitly set, prefer "sync"
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
        self.store.append_event(ev)
        step += 1

        # Stream updates and emit events by node update completion.
        # The concrete shape of updates is framework-specific; we record a summary.
        # Determine effective durability again for passing into stream()
        effective_durability = self.durability
        if effective_durability is None and thread_id:
            effective_durability = "sync"

        stream_kwargs: dict[str, Any] = {
            "stream_mode": list(self.stream_modes)
            if len(self.stream_modes) > 1
            else (self.stream_modes[0] if self.stream_modes else "updates")
        }
        if self.stream_subgraphs:
            stream_kwargs["subgraphs"] = True
        if effective_durability is not None:
            stream_kwargs["durability"] = effective_durability
        try:
            iterator = self.graph.stream(inputs, config or {}, **stream_kwargs)
        except TypeError:
            # some implementations may not accept subgraphs
            stream_kwargs.pop("subgraphs", None)
            iterator = self.graph.stream(inputs, config or {}, **stream_kwargs)

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
            # Persist chunks separately (optional) to keep main payload compact
            chunks_payload: dict[str, Any] = {"chunks": agg_chunks}
            chunks_b = self._normalize_bytes(chunks_payload)
            # Use STATE kind as a secondary blob for this step (no SNAPSHOT event is created here)
            chunks_ref = self.store.put_blob(self.run.run_id, step, BlobKind.STATE, chunks_b)
            payload: dict[str, Any] = {
                "message": {"content": "".join(agg_text)},
                "metadata": {"chunks_count": len(agg_chunks)},
                # Provide a reference to the token chunks blob for later inspection
                "chunks_ref": chunks_ref.model_dump(mode="json"),
            }
            payload_b2 = self._normalize_bytes(payload)
            out_blob2 = self.store.put_blob(self.run.run_id, step, BlobKind.OUTPUT, payload_b2)
            labels2 = dict(agg_labels)
            labels2["stream_mode"] = "messages"
            if agg_actor:
                labels2.setdefault("node", agg_actor)
            # Best-effort extraction of prompt/source messages for hashing and provider/model/params
            prompt_hash: str | None = None
            provider: str | None = None
            model: str | None = None
            params_meta: dict[str, Any] = {}
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
                    if provider is None and isinstance(meta.get("provider"), str):
                        provider = str(meta.get("provider"))
                    if model is None and isinstance(meta.get("model"), str):
                        model = str(meta.get("model"))
                    # Extract common LLM params when available
                    for p in ("temperature", "top_p", "tool_choice"):
                        val = meta.get(p)
                        if val is None and isinstance(meta.get("params"), dict):
                            params = meta["params"]
                            try:
                                val = params.get(p)
                            except Exception:
                                val = None
                        if val is not None and p not in params_meta:
                            try:
                                # keep only JSON-serializable scalars
                                if isinstance(val, str | int | float | bool):
                                    params_meta[p] = val
                            except Exception:
                                pass
                if sources:
                    prompt_hash = hash_bytes(_to_bytes({"sources": sources}))
            except Exception:
                prompt_hash = None
            # Compute anchor id for alignment
            try:
                anchor_id2 = self._make_anchor_id(ActionType.LLM, agg_actor or "graph", labels2)
                if anchor_id2:
                    labels2["anchor_id"] = anchor_id2
            except Exception:
                pass

            ev2 = Event(
                run_id=self.run.run_id,
                step=step,
                action_type=ActionType.LLM,
                actor=agg_actor or "graph",
                output_ref=out_blob2,
                hashes=(
                    {"output": out_blob2.sha256_hex}
                    | ({"prompt": prompt_hash} if prompt_hash else {})
                ),
                labels=labels2,
                model_meta={
                    "adapter_version": self.ADAPTER_VERSION,
                    "framework": "langgraph",
                    "chunks_count": len(agg_chunks),
                    **({"provider": provider} if provider else {}),
                    **({"model": model} if model else {}),
                    **params_meta,
                },
                ts=tw_now(),
            )
            # Merge a staged prompt hash when not present
            try:
                if "prompt" not in ev2.hashes:
                    staged = try_pop_prompt_hash()
                    if staged:
                        ev2 = ev2.model_copy(update={"hashes": {**ev2.hashes, "prompt": staged}})
            except Exception:
                pass
            self.store.append_event(ev2)
            step += 1
            # reset
            agg_key = None
            agg_chunks = []
            agg_text = []
            agg_labels = {}
            agg_actor = None

        for update in iterator:
            # Normalize stream shapes
            stream_mode_label: str | None = None
            namespace_label: str | None = None
            upd = update

            try:
                # (namespace, mode, data)
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
                # (mode, data)
                elif (
                    isinstance(update, tuple | list)
                    and len(update) == 2
                    and isinstance(update[0], str)
                ):
                    stream_mode_label = update[0]
                    upd = update[1]
                # (namespace, data)
                elif (
                    isinstance(update, tuple | list)
                    and len(update) == 2
                    and isinstance(update[0], tuple | list)
                ):
                    ns = [str(x) for x in update[0]]
                    namespace_label = "/".join(ns)
                    upd = update[1]
                else:
                    # single-mode guesses (values or updates)
                    stream_mode_label = single_mode_label
            except Exception:
                stream_mode_label = stream_mode_label or single_mode_label

            actor = self._infer_actor(upd)
            # Derive actor from namespace if not inferred
            if actor == "graph" and namespace_label:
                try:
                    last_seg = namespace_label.split("|")[-1]
                    actor = last_seg.split(":")[0] if ":" in last_seg else last_seg
                except Exception:
                    pass

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
                except Exception:
                    pass
                # Merge staged prompt hash when stream metadata is insufficient
                try:
                    if "prompt" not in ev_hashes:
                        staged_p = try_pop_prompt_hash()
                        if staged_p:
                            ev_hashes["prompt"] = staged_p
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
                output_ref=out_blob,
                hashes=ev_hashes,
                labels=ev_labels,
                model_meta=(base_meta if ev_meta is None else {**base_meta, **ev_meta}),
                ts=tw_now(),
            )
            if tool_meta:
                ev = ev.model_copy(update=tool_meta)

            self.store.append_event(ev)
            step += 1  # advance step after recording the update event
            updates_seen += 1
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
                            self.store.append_event(dec_ev)
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
                    hitl_blob = self.store.put_blob(
                        self.run.run_id,
                        step,
                        BlobKind.OUTPUT,
                        self._normalize_bytes(upd["__interrupt__"]),
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
                    self.store.append_event(hitl_ev)
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

        # Return best-effort result without re-executing the graph
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
        self.store.append_event(ev)

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
            if isinstance(update, dict):
                if "tool" in update or "tool_name" in update or update.get("tool_kind"):
                    return ActionType.TOOL
                # messages stream commonly includes token tuples or message metadata
                if "messages" in update or "llm_input_messages" in update:
                    return ActionType.LLM
        except Exception:
            pass
        return ActionType.SYS

    def _classify_tool_from_update(self, update: Any) -> dict[str, str] | None:
        # Allow user-provided classifier to recognize MCP calls (recommended)
        if self.tool_classifier and hasattr(update, "get"):
            # If the update contains a reference to the tool object
            tool_obj = update.get("tool") if isinstance(update, dict) else None
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
        if isinstance(update, dict):
            name = update.get("tool_name") or update.get("name")
            kind = update.get("tool_kind")
            if name and (kind == "MCP" or update.get("mcp_server") or update.get("mcp_transport")):
                out2: dict[str, str] = {
                    "tool_kind": str(kind or "MCP"),
                    "tool_name": str(name),
                }
                if update.get("mcp_server"):
                    out2["mcp_server"] = str(update.get("mcp_server"))
                if update.get("mcp_transport"):
                    out2["mcp_transport"] = str(update.get("mcp_transport"))
                return out2
            # Additionally inspect nested metadata produced by messages stream
            meta = update.get("metadata")
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
            if not isinstance(update, dict):
                return None
            if "args" in update or "kwargs" in update:
                args_v = update.get("args", [])
                kwargs_v = update.get("kwargs", {})
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
            ta = update.get("tool_args")
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
            inp = update.get("input")
            if isinstance(inp, dict):
                return {"args": [], "kwargs": inp}
            params = update.get("parameters")
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
