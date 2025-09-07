from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, cast
from uuid import UUID

from ..codec import from_bytes, to_bytes
from ..events import ActionType, BlobKind, Event
from ..store import LocalStore
from ..telemetry import replay_span_for_event
from .exceptions import MissingBlob, SchemaMismatch
from .langgraph import LangGraphReplayer, ReplaySession
from .wrappers import PlaybackLLM, PlaybackMemory, PlaybackTool


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
                from ..events import Event as _EventModel

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

        from ..bindings import bind_langgraph_playback as _bind

        def _installer(llm: PlaybackLLM, tool: PlaybackTool, memory: PlaybackMemory) -> None:
            llm.strict_meta = bool(strict_meta)
            tool.strict_meta = bool(strict_meta)
            _bind(graph=graph, llm=llm, tool=tool, memory=memory)

        replayer = LangGraphReplayer(graph=graph, store=store)
        return replayer.resume(
            run_id=run_id,
            from_step=from_step,
            thread_id=thread_id,
            install_wrappers=_installer,
            freeze_time=freeze_time,
        )
