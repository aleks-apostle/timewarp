from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID

from .codec import from_bytes, to_bytes
from .determinism import freeze_time_at
from .events import ActionType, BlobKind, Event, hash_bytes
from .store import LocalStore
from .telemetry import replay_span_for_event


class ReplayError(Exception):
    """Base class for replay-related errors."""


class MissingBlob(ReplayError):
    def __init__(self, *, run_id: UUID, step: int, kind: BlobKind, path: str) -> None:
        self.run_id = run_id
        self.step = step
        self.kind = kind
        self.path = path
        super().__init__(f"Missing blob for run={run_id} step={step} kind={kind.value} path={path}")


class SchemaMismatch(ReplayError):
    def __init__(self, message: str, *, step: int | None = None) -> None:
        self.step = step
        super().__init__(message)


class AdapterInvariant(ReplayError):
    def __init__(self, message: str, *, step: int | None = None) -> None:
        self.step = step
        super().__init__(message)


class MissingRecordedEvent(ReplayError):
    def __init__(self, *, run_id: UUID, after_step: int, action_type: ActionType) -> None:
        self.run_id = run_id
        self.after_step = after_step
        self.action_type = action_type
        super().__init__(
            f"Missing recorded {action_type.value} event after step={after_step} in run={run_id}"
        )


class LLMPromptMismatch(ReplayError):
    def __init__(self, step: int, *, expected_hash: str | None, got_hash: str | None) -> None:
        self.step = step
        self.expected_hash = expected_hash
        self.got_hash = got_hash
        super().__init__(
            f"LLM prompt mismatch at step={step}: expected={expected_hash} got={got_hash}"
        )


class ToolArgsMismatch(ReplayError):
    def __init__(self, step: int, *, expected_hash: str | None, got_hash: str | None) -> None:
        self.step = step
        self.expected_hash = expected_hash
        self.got_hash = got_hash
        super().__init__(
            f"TOOL args mismatch at step={step}: expected={expected_hash} got={got_hash}"
        )


class ToolsDigestMismatch(ReplayError):
    def __init__(self, step: int, *, expected_digest: str | None, got_digest: str | None) -> None:
        self.step = step
        self.expected_digest = expected_digest
        self.got_digest = got_digest
        super().__init__(
            f"LLM tools digest mismatch at step={step}: expected={expected_digest} got={got_digest}"
        )


class PromptContextMismatch(ReplayError):
    def __init__(self, step: int, *, expected_hash: str | None, got_hash: str | None) -> None:
        self.step = step
        self.expected_hash = expected_hash
        self.got_hash = got_hash
        super().__init__(
            f"LLM prompt_ctx mismatch at step={step}: expected={expected_hash} got={got_hash}"
        )


class RetrievalQueryMismatch(ReplayError):
    def __init__(self, step: int, *, expected_hash: str | None, got_hash: str | None) -> None:
        self.step = step
        self.expected_hash = expected_hash
        self.got_hash = got_hash
        super().__init__(
            f"RETRIEVAL query mismatch at step={step}: expected={expected_hash} got={got_hash}"
        )


class RetrievalPolicyMismatch(ReplayError):
    def __init__(
        self, step: int, *, field: str, expected: str | int | None, got: str | int | None
    ) -> None:
        self.step = step
        self.field = field
        self.expected = expected
        self.got = got
        super().__init__(
            f"RETRIEVAL policy mismatch at step={step} ({field}): expected={expected} got={got}"
        )


class ModelMetaMismatch(ReplayError):
    def __init__(self, step: int, *, diffs: list[str]) -> None:
        self.step = step
        self.diffs = diffs
        msg = ", ".join(diffs) if diffs else "model_meta mismatch"
        super().__init__(f"Model meta mismatch at step={step}: {msg}")


@dataclass
class _EventCursor:
    events: list[Event]
    action_type: ActionType
    start_index: int = 0
    thread_id: str | None = None

    def next(self) -> Event:
        for i in range(self.start_index, len(self.events)):
            e = self.events[i]
            if e.action_type is self.action_type:
                if self.thread_id is None or (e.labels.get("thread_id") == self.thread_id):
                    self.start_index = i + 1
                    return e
        raise MissingRecordedEvent(
            run_id=self.events[0].run_id if self.events else UUID(int=0),
            after_step=self.events[self.start_index - 1].step if self.start_index > 0 else -1,
            action_type=self.action_type,
        )


def _hash_prompt_like(prompt: Any, *, messages: Any | None = None) -> str:
    """Compute a stable sha256 hex over LLM prompt/messages like the recorder.

    Best-effort: if messages provided, prefer it; else hash the prompt itself.
    """
    try:
        obj: Any
        if messages is not None:
            obj = {"messages": messages}
        else:
            obj = {"prompt": prompt}
        return hash_bytes(to_bytes(obj))
    except Exception:
        # last resort
        return hash_bytes(to_bytes({"_repr": repr(prompt)}))


def _hash_tools_list(tools: Any) -> str:
    try:
        return hash_bytes(to_bytes({"tools": tools}))
    except Exception:
        return hash_bytes(to_bytes({"_repr": repr(tools)}))


@dataclass
class PlaybackLLM:
    store: LocalStore
    cursor: _EventCursor
    override: dict[int, Any] = field(default_factory=dict)
    strict_meta: bool = False
    freeze_time: bool = False

    def invoke(self, prompt: Any, **kwargs: Any) -> Any:
        ev = self.cursor.next()
        # Validate prompt hash if available on the recorded event
        recorded_prompt_hash: str | None = None
        try:
            recorded_prompt_hash = ev.hashes.get("prompt") if ev.hashes else None
        except Exception:
            recorded_prompt_hash = None
        if recorded_prompt_hash:
            # Attempt to extract messages-style input
            msgs = kwargs.get("messages")
            got_hash = _hash_prompt_like(prompt, messages=msgs)
            if got_hash != recorded_prompt_hash:
                raise LLMPromptMismatch(
                    ev.step, expected_hash=recorded_prompt_hash, got_hash=got_hash
                )
        # Optional tools digest validation (best-effort)
        try:
            recorded_tools_digest: str | None = None
            if ev.hashes:
                recorded_tools_digest = ev.hashes.get("tools")
            if recorded_tools_digest is None:
                recorded_tools_digest = ev.tools_digest
            observed_tools = (
                kwargs.get("tools")
                or kwargs.get("available_tools")
                or (
                    kwargs.get("_tw_model_meta", {}).get("tools")
                    if isinstance(kwargs.get("_tw_model_meta"), dict)
                    else None
                )
            )
            if observed_tools is not None and recorded_tools_digest is not None:
                got_td = _hash_tools_list(observed_tools)
                if got_td != recorded_tools_digest:
                    raise ToolsDigestMismatch(
                        ev.step, expected_digest=recorded_tools_digest, got_digest=got_td
                    )
        except ToolsDigestMismatch:
            raise
        except Exception:
            pass
        # Optional prompt_ctx validation when both messages and tools are available
        try:
            recorded_ctx: str | None = ev.hashes.get("prompt_ctx") if ev.hashes else None
            if recorded_ctx is not None:
                msgs2 = kwargs.get("messages")
                tools2 = (
                    kwargs.get("tools")
                    or kwargs.get("available_tools")
                    or (
                        kwargs.get("_tw_model_meta", {}).get("tools")
                        if isinstance(kwargs.get("_tw_model_meta"), dict)
                        else None
                    )
                )
                if msgs2 is not None and tools2 is not None:
                    got_ctx = hash_bytes(to_bytes({"messages": msgs2, "tools": tools2}))
                    if got_ctx != recorded_ctx:
                        raise PromptContextMismatch(
                            ev.step, expected_hash=recorded_ctx, got_hash=got_ctx
                        )
        except PromptContextMismatch:
            raise
        except Exception:
            pass
        # Optional model_meta validation (subset, opt-in)
        if self.strict_meta:
            try:
                observed = kwargs.get("_tw_model_meta")
                if isinstance(observed, dict):
                    recorded = ev.model_meta or {}
                    diffs: list[str] = _compare_model_meta(recorded, observed)
                    if diffs:
                        raise ModelMetaMismatch(ev.step, diffs=diffs)
            except ModelMetaMismatch:
                raise
            except Exception:
                # Best-effort; ignore meta errors if shapes are unexpected
                pass

        # One-shot override
        def _produce() -> Any:
            if ev.step in self.override:
                return self.override[ev.step]
            if not ev.output_ref:
                raise MissingBlob(
                    run_id=ev.run_id, step=ev.step, kind=BlobKind.OUTPUT, path="<none>"
                )
            raw = self.store.get_blob(ev.output_ref)
            return from_bytes(raw)

        if self.freeze_time:
            with freeze_time_at(ev.ts):
                return _produce()
        return _produce()


@dataclass
class PlaybackMemory:
    store: LocalStore
    retrieval_cursor: _EventCursor
    freeze_time: bool = False

    def retrieve(
        self,
        query: Any | None = None,
        *,
        retriever: str | None = None,
        top_k: int | None = None,
    ) -> list[Any] | Any:
        """Return recorded retrieval items for the next RETRIEVAL event.

        Validates query hash, retriever, and top_k when present on the recorded event.
        Returns the recorded payload's items list by default.
        """
        ev = self.retrieval_cursor.next()
        # Validate query hash when present
        try:
            expected_q = ev.hashes.get("query") if ev.hashes else None
            if expected_q is not None and query is not None:
                got_q = hash_bytes(to_bytes(query))
                if got_q != expected_q:
                    raise RetrievalQueryMismatch(ev.step, expected_hash=expected_q, got_hash=got_q)
        except RetrievalQueryMismatch:
            raise
        except Exception:
            pass
        # Validate simple policy fields if available
        try:
            if ev.top_k is not None and top_k is not None and int(ev.top_k) != int(top_k):
                raise RetrievalPolicyMismatch(ev.step, field="top_k", expected=ev.top_k, got=top_k)
            if (
                ev.retriever is not None
                and retriever is not None
                and str(ev.retriever) != str(retriever)
            ):
                raise RetrievalPolicyMismatch(
                    ev.step, field="retriever", expected=ev.retriever, got=retriever
                )
        except RetrievalPolicyMismatch:
            raise
        except Exception:
            pass

        # Produce items from recorded blob
        def _produce() -> list[Any] | Any:
            if not ev.output_ref:
                raise MissingBlob(
                    run_id=ev.run_id, step=ev.step, kind=BlobKind.MEMORY, path="<none>"
                )
            raw = self.store.get_blob(ev.output_ref)
            obj = from_bytes(raw)
            try:
                if isinstance(obj, dict) and isinstance(obj.get("items"), list):
                    return obj["items"]
            except Exception:
                pass
            return obj

        if self.freeze_time:
            with freeze_time_at(ev.ts):
                return _produce()
        return _produce()


@dataclass
class PlaybackTool:
    store: LocalStore
    cursor: _EventCursor
    override: dict[int, Any] = field(default_factory=dict)
    strict_meta: bool = False
    freeze_time: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # tool as callable
        ev = self.cursor.next()
        # Validate args hash if present
        expected: str | None = None
        try:
            expected = ev.hashes.get("args") if ev.hashes else None
        except Exception:
            expected = None
        if expected:
            got = hash_bytes(to_bytes({"args": args, "kwargs": kwargs}))
            if got != expected:
                raise ToolArgsMismatch(ev.step, expected_hash=expected, got_hash=got)

        def _produce() -> Any:
            if ev.step in self.override:
                return self.override[ev.step]
            if not ev.output_ref:
                raise MissingBlob(
                    run_id=ev.run_id, step=ev.step, kind=BlobKind.OUTPUT, path="<none>"
                )
            raw = self.store.get_blob(ev.output_ref)
            return from_bytes(raw)

        if self.freeze_time:
            with freeze_time_at(ev.ts):
                return _produce()
        return _produce()


def _compare_model_meta(recorded: dict[str, Any], observed: dict[str, Any]) -> list[str]:
    """Return a list of human-readable diffs for intersecting keys.

    We restrict to common, stable keys to avoid framework variance.
    """
    keys = {"provider", "model", "temperature", "top_p", "tool_choice"}
    diffs: list[str] = []
    for k in keys:
        if k in recorded and k in observed:
            rv = recorded.get(k)
            ov = observed.get(k)
            if rv != ov:
                diffs.append(f"{k}: recorded={rv!r} observed={ov!r}")
    return diffs


@dataclass
class ReplaySession:
    run_id: UUID
    checkpoint_id: str | None
    result: Any | None
    # Keep references to wrappers so callers can inspect/teardown if needed
    playback_llm: PlaybackLLM | None = None
    playback_tool: PlaybackTool | None = None

    def teardown(self) -> None:
        # No-op for now; placeholder for restoring monkeypatches
        return None


@dataclass
class LangGraphReplayer:
    graph: Any
    store: LocalStore

    def resume(
        self,
        run_id: UUID,
        from_step: int | None,
        thread_id: str | None,
        *,
        install_wrappers: Callable[[PlaybackLLM, PlaybackTool, PlaybackMemory], None] | None = None,
        freeze_time: bool = False,
    ) -> ReplaySession:
        events = self.store.list_events(run_id)
        # Find initial input
        inputs: Any | None = None
        for ev in events:
            if ev.input_ref is not None:
                inputs = from_bytes(self.store.get_blob(ev.input_ref))
                break
        # Compute nearest checkpoint_id from latest SNAPSHOT <= from_step
        target = from_step if from_step is not None else 10**9
        checkpoint_id: str | None = None
        if events:
            for ev in reversed(events):
                if ev.step <= target and ev.action_type is ActionType.SNAPSHOT:
                    if ev.labels and "checkpoint_id" in ev.labels:
                        checkpoint_id = ev.labels["checkpoint_id"]
                        break
        # Prepare playback wrappers
        llm_cursor = _EventCursor(
            events=events, action_type=ActionType.LLM, start_index=0, thread_id=thread_id
        )
        tool_cursor = _EventCursor(
            events=events, action_type=ActionType.TOOL, start_index=0, thread_id=thread_id
        )
        llm = PlaybackLLM(store=self.store, cursor=llm_cursor)
        tool = PlaybackTool(store=self.store, cursor=tool_cursor)
        mem_cursor = _EventCursor(
            events=events, action_type=ActionType.RETRIEVAL, start_index=0, thread_id=thread_id
        )
        memory = PlaybackMemory(store=self.store, retrieval_cursor=mem_cursor)
        llm.freeze_time = freeze_time
        tool.freeze_time = freeze_time
        memory.freeze_time = freeze_time
        # If caller provides installer, let them bind wrappers to the graph runtime
        if install_wrappers is None:
            # If there are any LLM/TOOL events, require wrappers to avoid live side-effects
            has_side_effects = any(
                e.action_type in (ActionType.LLM, ActionType.TOOL) for e in events
            )
            if has_side_effects:
                raise AdapterInvariant(
                    "Playback wrappers required: detected LLM/TOOL events in run. "
                    "Install adapters via `uv pip install -e .[adapters]` and bind with "
                    "`installers.bind_langgraph_playback(graph, llm, tool)`, or use the CLI "
                    "`resume` command which binds automatically."
                )
        else:
            install_wrappers(llm, tool, memory)
        # Execute graph from checkpoint using values-stream to advance deterministically
        result: Any | None = None
        cfg: dict[str, Any] = {"configurable": {}}
        if thread_id is not None:
            cfg["configurable"]["thread_id"] = thread_id
        if checkpoint_id is not None:
            cfg["configurable"]["checkpoint_id"] = checkpoint_id
        # Prefer .stream if available to consume values updates
        if hasattr(self.graph, "stream") and callable(self.graph.stream):
            iterator = cast(Iterable[Any], self.graph.stream(inputs, cfg, stream_mode=["values"]))
            for _ in iterator:
                pass
            # Best-effort final state via get_state
            try:
                get_state = getattr(self.graph, "get_state", None)
                if callable(get_state):
                    snapshot = get_state(cfg)
                    if isinstance(snapshot, dict) and "values" in snapshot:
                        result = snapshot["values"]
                    else:
                        result = snapshot
            except Exception:
                result = None
        elif hasattr(self.graph, "invoke") and callable(self.graph.invoke):
            result = self.graph.invoke(inputs, cfg)
        else:
            raise AdapterInvariant(
                "Graph does not support .stream or .invoke for replay. "
                "Ensure your app factory returns a compiled LangGraph or use the CLI `resume` "
                "which binds playback wrappers for you."
            )
        return ReplaySession(
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            result=result,
            playback_llm=llm,
            playback_tool=tool,
        )

    def fork_with_injection(
        self,
        run_id: UUID,
        at_step: int,
        replacement: Any,
        thread_id: str | None,
        *,
        install_wrappers: Callable[[PlaybackLLM, PlaybackTool], None] | None = None,
        freeze_time: bool = False,
    ) -> UUID:
        """Prepare a forked run by installing an override for a single LLM/TOOL event.

        Returns the new run_id. The actual recording of the forked run is expected to be
        performed by the caller's recorder during replay execution.
        """
        # Build wrappers with one-shot override
        events = self.store.list_events(run_id)
        llm_cursor = _EventCursor(
            events=events, action_type=ActionType.LLM, start_index=0, thread_id=thread_id
        )
        tool_cursor = _EventCursor(
            events=events, action_type=ActionType.TOOL, start_index=0, thread_id=thread_id
        )
        llm = PlaybackLLM(
            store=self.store,
            cursor=llm_cursor,
            override={at_step: replacement},
            freeze_time=freeze_time,
        )
        tool = PlaybackTool(
            store=self.store,
            cursor=tool_cursor,
            override={at_step: replacement},
            freeze_time=freeze_time,
        )
        if install_wrappers is None:
            raise AdapterInvariant("install_wrappers is required to bind overrides for forking")
        install_wrappers(llm, tool)
        # Create a forked Run with branch metadata for discoverability.
        # Attempt to copy basic metadata from the original Run.
        try:
            from .events import Run as _Run

            orig_run: _Run | None = None
            for r in self.store.list_runs():
                if r.run_id == run_id:
                    orig_run = r
                    break
            labels = {"branch_of": str(run_id)}
            new_run = _Run(
                project=orig_run.project if orig_run else None,
                name=(orig_run.name if orig_run else None),
                framework=(orig_run.framework if orig_run else None),
                code_version=(orig_run.code_version if orig_run else None),
                labels=labels,
            )
            self.store.create_run(new_run)
            return new_run.run_id
        except Exception:
            # Fallback: return a fresh UUID if creating Run fails
            from uuid import uuid4

            return uuid4()


@dataclass
class Replay:
    """In-memory replay session for a run.

    Provides stepwise navigation over recorded events with lightweight state inspection
    based on SNAPSHOT events. Output injection and skipping are applied as overlays
    (non-persistent) for exploratory debugging.
    """

    store: LocalStore
    run_id: UUID
    _events: list[Event] = field(init=False, default_factory=list)
    _pos: int = field(init=False, default=0)
    _overlay_outputs: dict[int, bytes] = field(init=False, default_factory=dict)
    _skipped: set[int] = field(init=False, default_factory=set)

    def __post_init__(self) -> None:
        self._events = self.store.list_events(self.run_id)
        # Version guards: ensure consistent schema_version and adapter_version within run
        if self._events:
            # schema_version consistency
            schema_versions = {e.schema_version for e in self._events}
            if len(schema_versions) > 1:
                raise SchemaMismatch(f"Mixed schema versions in run: {sorted(schema_versions)}")
            try:
                # Compare against current default schema version on Event model
                from .events import Event as _EventModel

                field_info = _EventModel.model_fields.get("schema_version")
                current = field_info.default if field_info is not None else None
                only = next(iter(schema_versions))
                if isinstance(current, int) and isinstance(only, int) and only != current:
                    raise SchemaMismatch(
                        f"Run schema_version={only} incompatible with current={current}"
                    )
            except Exception:
                # Best-effort; if reflection fails, skip strict check
                pass
            # adapter_version consistency when present
            adapter_versions: set[str] = set()
            for e in self._events:
                mm = e.model_meta or {}
                try:
                    av = mm.get("adapter_version") if isinstance(mm, dict) else None
                    if isinstance(av, str):
                        adapter_versions.add(av)
                except Exception:
                    continue
            if len(adapter_versions) > 1:
                raise SchemaMismatch(
                    f"Mixed adapter_version values in run: {sorted(adapter_versions)}"
                )
        self._pos = 0

    # --- navigation ---

    def goto(self, step: int) -> Replay:
        self._pos = 0
        for i, ev in enumerate(self._events):
            if ev.step >= step:
                self._pos = i
                break
        else:
            self._pos = len(self._events)
        return self

    def step(self) -> Replay:
        if self._pos < len(self._events):
            # Emit a replay span for the event being advanced
            ev = self._events[self._pos]
            with replay_span_for_event(ev):
                self._pos += 1
        return self

    def next(self, action_type: ActionType | None = None) -> Replay:
        if action_type is None:
            return self.step()
        for i in range(self._pos, len(self._events)):
            if self._events[i].action_type == action_type:
                with replay_span_for_event(self._events[i]):
                    self._pos = i + 1
                break
        else:
            self._pos = len(self._events)
        return self

    # --- overlays ---

    def inject(
        self, step: int, output: dict[str, Any] | list[Any] | str | int | float | bool | None
    ) -> None:
        from .codec import to_bytes

        self._overlay_outputs[step] = to_bytes(output)

    def skip(self, step: int) -> None:
        self._skipped.add(step)

    # --- inspection ---

    def current_event(self) -> Event | None:
        if self._pos == 0:
            return None
        return self._events[self._pos - 1]

    def inspect_state(self) -> dict[str, Any] | list[Any] | None:
        """Return the most recent SNAPSHOT or STATE blob <= current position.

        If an overlay injection exists for that step, return the overlay.
        """
        idx = self._pos - 1
        # First pass: prefer latest values-stream state if available
        j = idx
        while j >= 0:
            evv = self._events[j]
            if evv.labels.get("stream_mode") == "values" and evv.output_ref:
                try:
                    data_v = self.store.get_blob(evv.output_ref)
                    obj_v = from_bytes(data_v)
                    if isinstance(obj_v, dict | list):
                        return obj_v
                except Exception:
                    break
            j -= 1
        # Second pass: locate the last snapshot and reconstruct by applying subsequent updates
        snap_index: int | None = None
        base_state: dict[str, Any] | None = None
        while idx >= 0:
            ev = self._events[idx]
            if ev.action_type is ActionType.SNAPSHOT and ev.output_ref:
                if ev.step in self._overlay_outputs:
                    try:
                        obj = from_bytes(self._overlay_outputs[ev.step])
                    except Exception as e:  # pragma: no cover - defensive
                        raise SchemaMismatch("Overlay decode failed", step=ev.step) from e
                    if isinstance(obj, dict | list):
                        if isinstance(obj, list):
                            return obj
                        base_state = cast(dict[str, Any], obj)
                        snap_index = idx
                        break
                    raise SchemaMismatch(
                        "Overlay content is not structured JSON (dict/list)", step=ev.step
                    )
                try:
                    data = self.store.get_blob(ev.output_ref)
                except FileNotFoundError as e:
                    raise MissingBlob(
                        run_id=self.run_id,
                        step=ev.step,
                        kind=ev.output_ref.kind,
                        path=ev.output_ref.path,
                    ) from e
                except Exception as e:  # pragma: no cover - defensive
                    raise SchemaMismatch("Blob read failed", step=ev.step) from e
                try:
                    obj = from_bytes(data)
                except Exception as e:  # pragma: no cover - defensive
                    raise SchemaMismatch("Blob JSON decode failed", step=ev.step) from e
                if isinstance(obj, list):
                    return obj
                if isinstance(obj, dict):
                    base_state = cast(dict[str, Any], obj)
                    snap_index = idx
                    break
                # State-like blob but not JSON
                raise SchemaMismatch(
                    "State blob content is not structured JSON (dict/list)", step=ev.step
                )
            idx -= 1
        if base_state is None:
            return None
        # Apply updates after the snapshot up to current position
        state: dict[str, Any] = dict(base_state)

        def deep_merge(dst: dict[str, Any], patch: dict[str, Any]) -> None:
            for k, v in patch.items():
                if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
                    deep_merge(cast(dict[str, Any], dst[k]), cast(dict[str, Any], v))
                else:
                    dst[k] = v

        def extract_patch(obj: Any) -> dict[str, Any] | None:
            # Accept various shapes: {"values": {...}}, {"state": {...}}, or {node: {...}}
            if isinstance(obj, dict):
                v1 = obj.get("values")
                if isinstance(v1, dict):
                    return cast(dict[str, Any], v1)
                v2 = obj.get("state")
                if isinstance(v2, dict):
                    return cast(dict[str, Any], v2)
                if len(obj) == 1:
                    ((only_key, only_val),) = obj.items()
                    if isinstance(only_val, dict):
                        return cast(dict[str, Any], only_val)
            return None

        for k in range((snap_index or 0) + 1, self._pos):
            evu = self._events[k]
            if evu.step in self._skipped:
                continue
            # Ignore messages-mode during reconstruction
            if evu.labels.get("stream_mode") == "messages":
                continue
            # Read overlay or event blob
            raw: bytes | None = None
            if evu.step in self._overlay_outputs:
                raw = self._overlay_outputs[evu.step]
            elif evu.output_ref:
                try:
                    raw = self.store.get_blob(evu.output_ref)
                except Exception:
                    raw = None
            if raw is None:
                continue
            try:
                up = from_bytes(raw)
            except Exception:
                continue
            patch = extract_patch(up)
            if patch is not None:
                deep_merge(state, patch)
        return state

    # --- data access helpers ---

    def iter_timeline(self) -> Iterable[Event]:
        yield from self._events

    # --- utilities ---

    def snapshot_now(self) -> Event:
        """Compute current state via inspect_state and append a SNAPSHOT event.

        Returns the created Event. Raises SchemaMismatch if state cannot be serialized.
        """
        state = self.inspect_state()
        if not isinstance(state, dict | list):
            raise SchemaMismatch("Current state is not JSON-serializable (dict/list)")
        step = (self._events[-1].step + 1) if self._events else 0
        from .codec import to_bytes

        blob = self.store.put_blob(self.run_id, step, BlobKind.STATE, to_bytes(state))
        ev = Event(
            run_id=self.run_id,
            step=step,
            action_type=ActionType.SNAPSHOT,
            actor="debugger",
            output_ref=blob,
            hashes={"state": blob.sha256_hex},
        )
        self.store.append_event(ev)
        # Refresh internal events timeline and position remains unchanged
        self._events = self.store.list_events(self.run_id)
        return ev

    # --- convenience facade ---

    @staticmethod
    def resume(
        store: LocalStore,
        *,
        app_factory: str,
        run_id: UUID,
        from_step: int | None = None,
        thread_id: str | None = None,
        strict_meta: bool = False,
        freeze_time: bool = False,
    ) -> ReplaySession:
        """One-call resume for LangGraph runs using recorded outputs.

        Parameters
        - store: LocalStore instance
        - app_factory: "module:function" that returns a compiled LangGraph
        - run_id: run to resume
        - from_step/thread_id: optional resume cursor and LangGraph thread id
        - strict_meta: validate provider/model/params observed vs recorded
        - freeze_time: freeze time to recorded event timestamps during replay
        """
        from importlib import import_module

        mod_name, func_name = app_factory.split(":", 1)
        mod = import_module(mod_name)
        from collections.abc import Callable as _Callable
        from typing import Any as _Any
        from typing import cast as _cast

        factory = _cast(_Callable[[], _Any], getattr(mod, func_name))
        graph = factory()

        from .adapters import installers as _installers

        def _installer(llm: PlaybackLLM, tool: PlaybackTool, memory: PlaybackMemory) -> None:
            llm.strict_meta = bool(strict_meta)
            tool.strict_meta = bool(strict_meta)
            _installers.bind_langgraph_playback(graph, llm, tool, memory)

        replayer = LangGraphReplayer(graph=graph, store=store)
        return replayer.resume(
            run_id=run_id,
            from_step=from_step,
            thread_id=thread_id,
            install_wrappers=_installer,
            freeze_time=freeze_time,
        )
