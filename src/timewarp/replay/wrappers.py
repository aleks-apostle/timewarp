from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from ..codec import from_bytes, to_bytes
from ..determinism import freeze_time_at
from ..events import ActionType, BlobKind, Event, hash_bytes
from ..store import LocalStore
from ..utils.hashing import hash_prompt, hash_prompt_ctx, hash_tools_list
from ..utils.logging import log_warn_once
from .exceptions import (
    LLMPromptMismatch,
    MissingBlob,
    MissingRecordedEvent,
    ModelMetaMismatch,
    PromptContextMismatch,
    RetrievalPolicyMismatch,
    RetrievalQueryMismatch,
    ToolArgsMismatch,
    ToolsDigestMismatch,
)


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
class PlaybackLLM:
    store: LocalStore
    cursor: _EventCursor
    override: dict[int, Any] = field(default_factory=dict)
    strict_meta: bool = False
    freeze_time: bool = False
    # Optional per-agent prompt overrides. Keyed by labels.node or actor.
    # The callable receives the original messages/prompt and returns the transformed value.
    prompt_overrides: dict[str, Callable[[Any], Any]] = field(default_factory=dict)
    # When True, allow prompt/prompt_ctx mismatches (used by replay forks that intentionally
    # tweak prompts but keep outputs deterministic). Defaults to False (strict).
    allow_diff: bool = False

    def invoke(self, prompt: Any, **kwargs: Any) -> Any:
        ev = self.cursor.next()
        # Apply per-agent prompt override when configured for this event
        try:
            agent_key = ev.labels.get("node") if ev.labels else None
            if not agent_key:
                agent_key = ev.actor
            fn = None
            if isinstance(agent_key, str) and agent_key:
                fn = self.prompt_overrides.get(agent_key)
            if callable(fn):
                if "messages" in kwargs and kwargs.get("messages") is not None:
                    try:
                        kwargs["messages"] = fn(kwargs.get("messages"))
                    except Exception:
                        # Best-effort: if override fails, keep original
                        log_warn_once("replay.llm.prompt_override_failed")
                else:
                    try:
                        prompt = fn(prompt)
                    except Exception:
                        log_warn_once("replay.llm.prompt_override_failed")
        except Exception:
            # Do not let override plumbing break deterministic playback
            log_warn_once("replay.llm.prompt_override_outer_failed")

        # Validate prompt hash if available on the recorded event (post-override)
        recorded_prompt_hash: str | None = None
        try:
            recorded_prompt_hash = ev.hashes.get("prompt") if ev.hashes else None
        except Exception:
            recorded_prompt_hash = None
        if recorded_prompt_hash:
            # Attempt to extract messages-style input
            msgs = kwargs.get("messages")
            got_hash = hash_prompt(messages=msgs, prompt=prompt)
            if got_hash != recorded_prompt_hash and not self.allow_diff:
                raise LLMPromptMismatch(
                    ev.step, expected_hash=recorded_prompt_hash, got_hash=got_hash
                )
            # When we allow diffs, opportunistically stage the new hash for branch recording
            if got_hash and got_hash != recorded_prompt_hash and self.allow_diff:
                try:
                    # Only import locally to avoid hard dependency for non-recording paths
                    from ..bindings import (
                        stage_prompt_hash as _stage_prompt_hash,
                    )

                    _stage_prompt_hash(got_hash)
                except Exception:
                    # best-effort
                    log_warn_once("replay.llm.stage_prompt_hash_failed")
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
                got_td = hash_tools_list(observed_tools)
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
                    got_ctx = hash_prompt_ctx(messages=msgs2, tools=tools2)
                    if got_ctx != recorded_ctx and not self.allow_diff:
                        raise PromptContextMismatch(
                            ev.step, expected_hash=recorded_ctx, got_hash=got_ctx
                        )
                    if got_ctx and got_ctx != recorded_ctx and self.allow_diff:
                        try:
                            from ..bindings import (
                                stage_prompt_hash as _stage_prompt_hash,
                            )

                            # For prompt_ctx, also stage a prompt hash derived from messages only
                            # so that recorders that pop staged hashes can pick it up.
                            _stage_prompt_hash(hash_prompt(messages=msgs2, prompt=None))
                        except Exception:
                            log_warn_once("replay.llm.stage_prompt_ctx_hash_failed")
        except PromptContextMismatch:
            raise
        except Exception:
            log_warn_once("replay.llm.prompt_ctx_validation_failed")
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
                log_warn_once("replay.llm.model_meta_compare_failed")

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
