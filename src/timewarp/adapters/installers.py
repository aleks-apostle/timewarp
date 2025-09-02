from __future__ import annotations

# Installer helpers to bind Playback wrappers and record-taps into common runtimes.
#
# Playback mode patches LangChain core:
# - Patch BaseChatModel/BaseLanguageModel.invoke to route through PlaybackLLM
# - Patch BaseTool.__call__ to route through PlaybackTool
#
# Record-tap mode computes prompt/args hashes at call sites and stages them for the
# recorder to merge into the next LLM/TOOL event (robust determinism checks).
#
# These patches are global within the Python process. Callers should prefer
# using them in short-lived CLI/debug sessions. A teardown callable is returned
# to restore the original methods.
from collections import deque
from collections.abc import Callable
from typing import Any

from ..codec import to_bytes
from ..events import hash_bytes
from ..replay import PlaybackLLM, PlaybackTool

# --- Staging queues for record-time taps (process-local, FIFO) ---
_STAGED_PROMPTS: deque[str] = deque()
_STAGED_TOOLARGS: deque[str] = deque()


def stage_prompt_hash(h: str) -> None:
    _STAGED_PROMPTS.append(h)


def stage_tool_args_hash(h: str) -> None:
    _STAGED_TOOLARGS.append(h)


def try_pop_prompt_hash() -> str | None:
    try:
        return _STAGED_PROMPTS.popleft()
    except Exception:
        return None


def try_pop_tool_args_hash() -> str | None:
    try:
        return _STAGED_TOOLARGS.popleft()
    except Exception:
        return None


def bind_langgraph_playback(graph: Any, llm: PlaybackLLM, tool: PlaybackTool) -> Callable[[], None]:
    """Bind playback wrappers to common integration points.

    Parameters
    - graph: compiled LangGraph (unused for now; reserved for future binding)
    - llm: PlaybackLLM instance constructed by the replayer
    - tool: PlaybackTool instance constructed by the replayer

    Returns
    - teardown() callable that restores any monkeypatched methods.

    Notes
    - This function performs best-effort patching and silently skips when
      optional dependencies are not installed.
    - Patching is intentionally minimal to avoid overreaching:
      * We patch Chat/Language model .invoke at the base class in LangChain core.
      * We patch BaseTool.__call__ to intercept tool executions.
    """

    teardowns: list[Callable[[], None]] = []

    # Patch LangChain Chat/Language models (import lazily to avoid mypy issues)
    import importlib

    BLM: Any = None
    try:  # langchain_core >= 0.3
        _m = importlib.import_module("langchain_core.language_models.base")
        BLM = getattr(_m, "BaseLanguageModel", None)
    except Exception:  # pragma: no cover - optional dep
        BLM = None

    BCM: Any = None
    try:
        _m2 = importlib.import_module("langchain_core.language_models.chat_models")
        BCM = getattr(_m2, "BaseChatModel", None)
    except Exception:  # pragma: no cover - optional dep
        BCM = None

    def _patch_invoke_on(cls: Any) -> Callable[[], None] | None:
        if cls is None:
            return None
        # Guard: ensure attribute exists
        if not hasattr(cls, "invoke"):
            return None

        orig_invoke = cls.invoke

        def _patched_invoke(self: Any, prompt: Any, **kwargs: Any) -> Any:
            # Pass through observed model meta for strict validation (if enabled)
            meta = _extract_model_meta(self)
            # Avoid clobbering if caller provided one
            if "_tw_model_meta" not in kwargs and meta:
                kwargs["_tw_model_meta"] = meta
            return llm.invoke(prompt, **kwargs)

        cls.invoke = _patched_invoke

        def _undo() -> None:
            try:
                cls.invoke = orig_invoke
            except Exception:
                pass

        return _undo

    for base in (BCM, BLM):
        undo = _patch_invoke_on(base)
        if undo is not None:
            teardowns.append(undo)

    # Patch LangChain BaseTool.__call__ to route through PlaybackTool
    BaseToolCls: Any = None
    try:
        _mt = importlib.import_module("langchain_core.tools.base")
        BaseToolCls = getattr(_mt, "BaseTool", None)
    except Exception:  # pragma: no cover - optional dep
        BaseToolCls = None

    if BaseToolCls is not None and callable(BaseToolCls):
        orig_call = BaseToolCls.__call__

        def _patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
            return tool(*args, **kwargs)

        BaseToolCls.__call__ = _patched_call

        def _undo_tool() -> None:
            try:
                BaseToolCls.__call__ = orig_call
            except Exception:
                pass

        teardowns.append(_undo_tool)

    # Compose teardowns
    def teardown() -> None:
        for f in reversed(teardowns):
            try:
                f()
            except Exception:
                continue

    return teardown


def bind_langgraph_record() -> Callable[[], None]:
    """Install record-time taps for LangChain core to compute prompt/args hashes.

    Returns a teardown() callable to restore original methods.
    No-ops when optional dependencies are absent.
    """
    teardowns: list[Callable[[], None]] = []

    import importlib

    # Patch Chat/Language model invoke to compute prompt/messages hash
    BLM: Any = None
    BCM: Any = None
    try:
        _m = importlib.import_module("langchain_core.language_models.base")
        BLM = getattr(_m, "BaseLanguageModel", None)
    except Exception:  # pragma: no cover - optional
        BLM = None
    try:
        _m2 = importlib.import_module("langchain_core.language_models.chat_models")
        BCM = getattr(_m2, "BaseChatModel", None)
    except Exception:  # pragma: no cover - optional
        BCM = None

    def _patch_llm_on(cls: Any) -> Callable[[], None] | None:
        if cls is None or not hasattr(cls, "invoke"):
            return None
        orig = cls.invoke

        def _patched(self: Any, prompt: Any, **kwargs: Any) -> Any:
            # Prefer explicit messages if provided; else hash prompt
            try:
                msgs = kwargs.get("messages")
                if msgs is not None:
                    stage_prompt_hash(hash_bytes(to_bytes({"messages": msgs})))
                else:
                    stage_prompt_hash(hash_bytes(to_bytes({"prompt": prompt})))
            except Exception:
                # best-effort; failure to hash should not break runtime
                pass
            return orig(self, prompt, **kwargs)

        cls.invoke = _patched

        def _undo() -> None:
            try:
                cls.invoke = orig
            except Exception:
                pass

        return _undo

    for base in (BCM, BLM):
        undo = _patch_llm_on(base)
        if undo is not None:
            teardowns.append(undo)

    # Patch BaseTool.__call__ to compute args/kwargs hash
    BaseToolCls: Any = None
    try:
        _mt = importlib.import_module("langchain_core.tools.base")
        BaseToolCls = getattr(_mt, "BaseTool", None)
    except Exception:  # pragma: no cover - optional
        BaseToolCls = None

    if BaseToolCls is not None and callable(BaseToolCls):
        orig_call = BaseToolCls.__call__

        def _patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                stage_tool_args_hash(hash_bytes(to_bytes({"args": list(args), "kwargs": kwargs})))
            except Exception:
                pass
            return orig_call(self, *args, **kwargs)

        BaseToolCls.__call__ = _patched_call

        def _undo_tool() -> None:
            try:
                BaseToolCls.__call__ = orig_call
            except Exception:
                pass

        teardowns.append(_undo_tool)

    def teardown() -> None:
        for f in reversed(teardowns):
            try:
                f()
            except Exception:
                continue

    return teardown


def _extract_model_meta(model_obj: Any) -> dict[str, Any]:
    """Best-effort extraction of model meta from LangChain model instances.

    Looks for common attribute names and returns a small JSON-serializable dict.
    """
    out: dict[str, Any] = {}
    try:
        # Provider/model
        for key in ("provider", "_provider"):
            v = getattr(model_obj, key, None)
            if isinstance(v, str):
                out.setdefault("provider", v)
        for key in ("model", "model_name", "model_id"):
            v = getattr(model_obj, key, None)
            if isinstance(v, str):
                out.setdefault("model", v)
        # Params
        for key in ("temperature", "top_p", "tool_choice"):
            v = getattr(model_obj, key, None)
            if isinstance(v, str | int | float | bool):
                out[key] = v
    except Exception:
        return out
    return out
