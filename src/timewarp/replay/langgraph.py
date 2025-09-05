from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast
from uuid import UUID

from ..codec import from_bytes
from ..events import ActionType
from ..store import LocalStore
from .exceptions import AdapterInvariant
from .wrappers import PlaybackLLM, PlaybackMemory, PlaybackTool, _EventCursor


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
        install_wrappers: Callable[[PlaybackLLM, PlaybackTool, PlaybackMemory], None] | None = None,
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
        # Build playback memory wrapper for recorded retrieval events (for provider patching)
        mem_cursor = _EventCursor(
            events=events, action_type=ActionType.RETRIEVAL, start_index=0, thread_id=thread_id
        )
        memory = PlaybackMemory(store=self.store, retrieval_cursor=mem_cursor)
        memory.freeze_time = freeze_time
        if install_wrappers is None:
            raise AdapterInvariant("install_wrappers is required to bind overrides for forking")
        install_wrappers(llm, tool, memory)
        # Create a forked Run with branch metadata for discoverability.
        # Attempt to copy basic metadata from the original Run.
        try:
            from ..events import Run as _Run

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
