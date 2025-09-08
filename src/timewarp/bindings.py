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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from .codec import to_bytes
from .events import hash_bytes
from .replay import PlaybackLLM, PlaybackMemory, PlaybackTool
from .utils.logging import log_warn_once

# --- Staging queues for record-time taps ---
# New: session-scoped staging via ContextVar to avoid cross-run leakage.


@dataclass
class _RecordingSession:
    run_id: UUID
    prompts: deque[str] = field(default_factory=deque)
    toolargs: deque[str] = field(default_factory=deque)
    memtaps: deque[dict[str, Any]] = field(default_factory=deque)


_SESSION: ContextVar[_RecordingSession | None] = ContextVar("tw_recording_session", default=None)

# Legacy global fallbacks removed: require an active recording session


def begin_recording_session(run_id: UUID) -> Callable[[], None]:
    """Begin a recording session bound to the given run_id.

    Returns a teardown callable that restores the previous session.
    """

    sess = _RecordingSession(run_id=run_id)
    token: Token[_RecordingSession | None] = _SESSION.set(sess)

    def _end() -> None:
        try:
            _SESSION.reset(token)
        except Exception as e:
            # best-effort; log once to aid debugging of session scope reset
            log_warn_once("installers.session.reset_failed", e)

    return _end


def stage_prompt_hash(h: str) -> None:
    sess = _SESSION.get()
    if sess is None:
        return
    sess.prompts.append(h)


def stage_tool_args_hash(h: str) -> None:
    sess = _SESSION.get()
    if sess is None:
        return
    sess.toolargs.append(h)


def try_pop_prompt_hash() -> str | None:
    sess = _SESSION.get()
    if sess is None:
        return None
    try:
        return sess.prompts.popleft()
    except Exception:
        return None


def try_pop_tool_args_hash() -> str | None:
    sess = _SESSION.get()
    if sess is None:
        return None
    try:
        return sess.toolargs.popleft()
    except Exception:
        return None


@contextmanager
def record_taps(run_id: UUID) -> Iterator[None]:
    """Context manager that scopes record-time taps to a run_id.

    Begins a recording session and installs prompt/tool-args hash taps; restores on exit.
    """
    end_session = begin_recording_session(run_id)
    undo = bind_langgraph_record()
    try:
        yield None
    finally:
        try:
            undo()
        except Exception as e:
            log_warn_once("installers.record_taps.undo_failed", e)
        try:
            end_session()
        except Exception as e:
            log_warn_once("installers.record_taps.end_session_failed", e)


# --- Generic patching helpers / small utilities ---


def _patch_method(
    cls: Any, name: str, make_wrapper: Callable[[Callable[..., Any]], Callable[..., Any]]
) -> Callable[[], None] | None:
    """Patch a method on a class and return an undo callable.

    - No-ops when the attribute is missing or not callable.
    - `make_wrapper` receives the original function and must return the patched one.
    """
    if cls is None or not hasattr(cls, name):
        return None
    orig = getattr(cls, name)
    if not callable(orig):
        return None
    try:
        patched = make_wrapper(orig)
    except Exception as e:  # pragma: no cover - defensive
        log_warn_once("installers.patch.make_wrapper_failed", e)
        return None
    setattr(cls, name, patched)

    def _undo() -> None:
        try:
            setattr(cls, name, orig)
        except Exception as e:  # pragma: no cover - defensive
            log_warn_once("installers.patch.undo_failed", e)

    return _undo


def _extract_top_k_li(obj: Any, kwargs: dict[str, Any]) -> int | None:
    """Best-effort extraction of top_k from LlamaIndex-style retrievers.

    Checks kwargs (top_k/k) and object attributes (k/top_k).
    """
    try:
        if isinstance(kwargs.get("top_k"), int):
            return int(kwargs["top_k"])
    except Exception:
        pass
    try:
        if isinstance(kwargs.get("k"), int):
            return int(kwargs["k"])
    except Exception:
        pass
    try:
        k = getattr(obj, "k", None) or getattr(obj, "top_k", None)
        if isinstance(k, int):
            return int(k)
    except Exception:
        pass
    return None


def bind_langgraph_playback(
    graph: Any,
    llm: PlaybackLLM,
    tool: PlaybackTool,
    memory: PlaybackMemory | None = None,
    *,
    prompt_overrides: dict[str, Callable[[Any], Any]] | None = None,
) -> Callable[[], None]:
    """Bind playback wrappers to common integration points.

    Parameters
    - graph: compiled LangGraph (unused for now; reserved for future binding)
    - llm: PlaybackLLM instance constructed by the replayer
    - tool: PlaybackTool instance constructed by the replayer
    - memory: Optional PlaybackMemory for provider memory/retriever patching

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

    # Thread prompt_overrides into the LLM wrapper when provided
    try:
        if isinstance(prompt_overrides, dict) and prompt_overrides:
            llm.prompt_overrides = dict(prompt_overrides)
    except Exception as e:
        log_warn_once("installers.playback.prompt_overrides_failed", e)

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

    def _make_playback_invoke_wrapper(_orig: Callable[..., Any]) -> Callable[..., Any]:
        def _patched_invoke(self: Any, prompt: Any, **kwargs: Any) -> Any:
            # Pass through observed model meta for strict validation (if enabled)
            meta = _extract_model_meta(self)
            if "_tw_model_meta" not in kwargs and meta:
                kwargs["_tw_model_meta"] = meta
            return llm.invoke(prompt, **kwargs)

        return _patched_invoke

    for base in (BCM, BLM):
        undo = _patch_method(base, "invoke", _make_playback_invoke_wrapper)
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

        def _make_tool_call_wrapper(_orig: Callable[..., Any]) -> Callable[..., Any]:
            def _patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
                return tool(*args, **kwargs)

            return _patched_call

        undo_tool = _patch_method(BaseToolCls, "__call__", _make_tool_call_wrapper)
        if undo_tool is not None:
            teardowns.append(undo_tool)

    # --- Optional: patch common memory providers to route to PlaybackMemory ---
    if memory is not None:
        import importlib

        # Mem0 patchers
        try:  # pragma: no cover - optional dependency
            mem0_mod = importlib.import_module("mem0")
            cand = getattr(mem0_mod, "Memory", None) or getattr(mem0_mod, "Client", None)
            if cand is not None:
                cls = cand

                def _mk_mem0_playback_wrapper(
                    name: str,
                ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                    def _make(_orig: Callable[..., Any]) -> Callable[..., Any]:
                        def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                            # Avoid calling original to prevent side-effects / network
                            if name in ("search", "retrieve", "query"):
                                q = args[0] if args else kwargs.get("query")
                                tk = kwargs.get("top_k")
                                return memory.retrieve(q, retriever="mem0", top_k=tk)
                            if name in ("save", "add", "delete", "update"):
                                return None
                            return None

                        return _patched

                    return _make

                for m in ("search", "retrieve", "query", "save", "add", "delete", "update"):
                    undo = _patch_method(cls, m, _mk_mem0_playback_wrapper(m))
                    if undo is not None:
                        teardowns.append(undo)
        except Exception as e:
            log_warn_once("installers.mem0.patch_failed", e)

        # LlamaIndex retriever patch (best-effort)
        try:  # pragma: no cover - optional dependency
            li_mod = importlib.import_module("llama_index")
            RetrieverBase = getattr(li_mod, "BaseRetriever", None)
            if RetrieverBase is not None and hasattr(RetrieverBase, "retrieve"):

                def _make_li_playback_wrapper(_orig: Callable[..., Any]) -> Callable[..., Any]:
                    def _patched_retrieve(self: Any, *args: Any, **kwargs: Any) -> Any:
                        q = args[0] if args else kwargs.get("query")
                        items = memory.retrieve(
                            q, retriever="llamaindex", top_k=_extract_top_k_li(self, kwargs)
                        )
                        return items

                    return _patched_retrieve

                undo_li = _patch_method(RetrieverBase, "retrieve", _make_li_playback_wrapper)
                if undo_li is not None:
                    teardowns.append(undo_li)
        except Exception as e:
            log_warn_once("installers.llamaindex.patch_failed", e)

    # Compose teardowns
    def teardown() -> None:
        for f in reversed(teardowns):
            try:
                f()
            except Exception as e:
                log_warn_once("installers.bind_playback.teardown_failed", e)

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

    def _make_record_invoke_wrapper(orig: Callable[..., Any]) -> Callable[..., Any]:
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

        return _patched

    for base in (BCM, BLM):
        undo = _patch_method(base, "invoke", _make_record_invoke_wrapper)
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

        def _make_tool_record_wrapper(orig: Callable[..., Any]) -> Callable[..., Any]:
            def _patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
                try:
                    stage_tool_args_hash(
                        hash_bytes(to_bytes({"args": list(args), "kwargs": kwargs}))
                    )
                except Exception as e:
                    # best-effort; warn once for serialization failures
                    log_warn_once("installers.bind_record.stage_tool_args_failed", e)
                return orig(self, *args, **kwargs)

            return _patched_call

        undo_tool = _patch_method(BaseToolCls, "__call__", _make_tool_record_wrapper)
        if undo_tool is not None:
            teardowns.append(undo_tool)

    def teardown() -> None:
        for f in reversed(teardowns):
            try:
                f()
            except Exception as e:
                log_warn_once("installers.bind_record.teardown_failed", e)

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
    # Best-effort library versions enrichment
    try:
        from timewarp import __version__ as _tw_version  # local import

        out.setdefault("timewarp_version", _tw_version)
    except Exception:
        pass
    try:  # optional deps
        import importlib

        lg = importlib.import_module("langgraph")
        v = getattr(lg, "__version__", None)
        if isinstance(v, str):
            out.setdefault("langgraph_version", v)
    except Exception:
        pass
    try:
        import importlib

        lcc = importlib.import_module("langchain_core")
        v = getattr(lcc, "__version__", None)
        if isinstance(v, str):
            out.setdefault("langchain_core_version", v)
    except Exception:
        pass
    return out


# --- Provider memory taps (record mode) ---


def stage_memory_tap(env: dict[str, Any]) -> None:
    """Stage a normalized provider-tap envelope for persistence by the recorder.

    Expected shapes (minimal):
    - MEMORY write: {"kind": "MEMORY", "mem_provider": str, "mem_op": str,
                     "key": str, "value": Any, "mem_scope"?: str, "mem_space"?: str}
    - RETRIEVAL read: {"kind": "RETRIEVAL", "mem_provider": str, "query": Any,
                       "items": list[dict], "policy": {"retriever"?: str, "top_k"?: int},
                       "query_id"?: str}
    Unknown shapes are ignored.
    """
    try:
        kind = env.get("kind")
        if kind not in ("MEMORY", "RETRIEVAL"):
            return
        payload = dict(env)  # shallow copy
        sess = _SESSION.get()
        if sess is None:
            return
        sess.memtaps.append(payload)
    except Exception:
        # best-effort
        return


def try_pop_memory_taps(max_items: int = 1000) -> list[dict[str, Any]]:
    """Drain staged provider-tap envelopes (FIFO).

    Recorder will call this after each update to persist provider-origin events
    alongside core events, preserving causality in step ordering.
    """
    out: list[dict[str, Any]] = []
    sess = _SESSION.get()
    if sess is None:
        return out
    try:
        n = 0
        while sess.memtaps and n < max_items:
            out.append(sess.memtaps.popleft())
            n += 1
    except Exception as e:
        log_warn_once("installers.mem0.bind_failed", e)
    return out


def bind_memory_taps() -> Callable[[], None]:
    """Best-effort patchers for common memory providers to stage MEMORY/RETRIEVAL taps.

    This scaffolding intentionally no-ops when optional dependencies are not present.
    Concrete Mem0 and LlamaIndex patchers can be added incrementally.
    """
    teardowns: list[Callable[[], None]] = []

    import importlib

    # --- Mem0 (optional) ---
    try:  # pragma: no cover - optional dependency
        mem0_mod = importlib.import_module("mem0")
        # Heuristic: detect a Memory-like class with save/search/delete
        cand = getattr(mem0_mod, "Memory", None) or getattr(mem0_mod, "Client", None)
        if cand is not None:
            cls = cand

            def _mk_mem0_tap_wrapper(
                name: str,
            ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
                def _make(orig: Callable[..., Any]) -> Callable[..., Any]:
                    def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
                        res = orig(self, *args, **kwargs)
                        try:
                            if name in ("save", "add"):
                                stage_memory_tap(
                                    {
                                        "kind": "MEMORY",
                                        "mem_provider": "Mem0",
                                        "mem_op": "PUT",
                                        "key": kwargs.get("key") or "mem0",
                                        "value": args[0] if args else kwargs.get("item"),
                                    }
                                )
                            elif name in ("delete",):
                                stage_memory_tap(
                                    {
                                        "kind": "MEMORY",
                                        "mem_provider": "Mem0",
                                        "mem_op": "DELETE",
                                        "key": kwargs.get("key") or "mem0",
                                        "value": None,
                                    }
                                )
                            elif name in ("search", "retrieve", "query"):
                                q = args[0] if args else kwargs.get("query")
                                items = res if isinstance(res, list) else []
                                stage_memory_tap(
                                    {
                                        "kind": "RETRIEVAL",
                                        "mem_provider": "Mem0",
                                        "query": q,
                                        "items": items,
                                        "policy": {
                                            "retriever": "vector",
                                            "top_k": kwargs.get("top_k"),
                                        },
                                    }
                                )
                        except Exception:
                            pass
                        return res

                    return _patched

                return _make

            for m in ("save", "add", "delete", "search", "retrieve", "query"):
                undo = _patch_method(cls, m, _mk_mem0_tap_wrapper(m))
                if undo is not None:
                    teardowns.append(undo)
    except Exception:
        pass

    # --- LangChain Retrievers (optional) ---
    try:  # pragma: no cover - optional dependency
        import importlib

        lc_ret_mod = importlib.import_module("langchain_core.retrievers")
        BaseRetriever = getattr(lc_ret_mod, "BaseRetriever", None)
        if BaseRetriever is None:
            # Older/newer path
            lc_ret_mod2 = importlib.import_module("langchain_core.retrievers.base")
            BaseRetriever = getattr(lc_ret_mod2, "BaseRetriever", None)
    except Exception:
        BaseRetriever = None

    def _normalize_doc(x: Any) -> Any:
        try:
            # Prefer Pydantic v2 API when available
            if hasattr(x, "model_dump") and callable(x.model_dump):
                return x.model_dump()
        except Exception:
            pass
        try:
            # Fallback for Pydantic v1 models
            if hasattr(x, "dict") and callable(x.dict):
                return x.dict()
        except Exception:
            pass
        try:
            # Common LangChain Document shape
            pc = getattr(x, "page_content", None)
            if pc is not None:
                return {"page_content": pc, "metadata": getattr(x, "metadata", None)}
        except Exception:
            pass
        if isinstance(x, str | int | float | bool | dict | list):
            return x
        return {"_repr": repr(x)}

    def _extract_top_k(self: Any, kwargs: dict[str, Any]) -> int | None:
        try:
            if isinstance(kwargs.get("top_k"), int):
                return int(kwargs["top_k"])
        except Exception:
            pass
        try:
            k = getattr(self, "k", None) or getattr(self, "top_k", None)
            if isinstance(k, int):
                return k
        except Exception:
            pass
        try:
            sk = getattr(self, "search_kwargs", None)
            if isinstance(sk, dict) and isinstance(sk.get("k"), int):
                return int(sk["k"])
        except Exception:
            pass
        return None

    if BaseRetriever is not None and hasattr(BaseRetriever, "get_relevant_documents"):
        orig_get = BaseRetriever.get_relevant_documents

        def _patched_get(self: Any, query: Any, *args: Any, **kwargs: Any) -> Any:
            res = orig_get(self, query, *args, **kwargs)
            try:
                items = [_normalize_doc(xi) for xi in (list(res) if isinstance(res, list) else [])]
                stage_memory_tap(
                    {
                        "kind": "RETRIEVAL",
                        "mem_provider": f"LangChainRetriever:{type(self).__name__}",
                        "query": query,
                        "items": items,
                        "policy": {
                            "retriever": "vector",
                            "top_k": _extract_top_k(self, kwargs),
                        },
                    }
                )
            except Exception:
                pass
            return res

        BaseRetriever.get_relevant_documents = _patched_get

        def _undo_lc_ret_get() -> None:
            try:
                BaseRetriever.get_relevant_documents = orig_get
            except Exception as e:
                log_warn_once("installers.langchain_retriever.undo_get_failed", e)

        teardowns.append(_undo_lc_ret_get)

    # Some retrievers use .invoke as Runnable interface; patch to let .get_relevant_documents stage.
    if BaseRetriever is not None and hasattr(BaseRetriever, "invoke"):
        orig_invoke = BaseRetriever.invoke

        def _patched_invoke(self: Any, query: Any, *args: Any, **kwargs: Any) -> Any:
            # Call original; staging handled by get_relevant_documents internally.
            return orig_invoke(self, query, *args, **kwargs)

        BaseRetriever.invoke = _patched_invoke

        def _undo_lc_ret_invoke() -> None:
            try:
                BaseRetriever.invoke = orig_invoke
            except Exception as e:
                log_warn_once("installers.langchain_retriever.undo_invoke_failed", e)

        teardowns.append(_undo_lc_ret_invoke)

    # --- LlamaIndex (optional) ---
    try:  # pragma: no cover - optional dependency
        li_mod = importlib.import_module("llama_index")
        # Patch common retriever base class if present
        RetrieverBase = getattr(li_mod, "BaseRetriever", None)
        if RetrieverBase is not None and hasattr(RetrieverBase, "retrieve"):
            orig_ret = RetrieverBase.retrieve

            def _patched_retrieve(self: Any, *args: Any, **kwargs: Any) -> Any:
                res = orig_ret(self, *args, **kwargs)
                try:
                    q = args[0] if args else kwargs.get("query")
                    items = (
                        [
                            (
                                xi.model_dump()
                                if hasattr(xi, "model_dump") and callable(xi.model_dump)
                                else (
                                    xi.dict() if hasattr(xi, "dict") and callable(xi.dict) else xi
                                )
                            )
                            for xi in res
                        ]
                        if isinstance(res, list)
                        else []
                    )
                    stage_memory_tap(
                        {
                            "kind": "RETRIEVAL",
                            "mem_provider": "LlamaIndex",
                            "query": q,
                            "items": items,
                            "policy": {
                                "retriever": "llamaindex",
                                "top_k": _extract_top_k_li(self, kwargs),
                            },
                        }
                    )
                except Exception:
                    pass
                return res

            RetrieverBase.retrieve = _patched_retrieve

            def _undo_li() -> None:
                try:
                    RetrieverBase.retrieve = orig_ret
                except Exception:
                    pass

            teardowns.append(_undo_li)
    except Exception as e:
        log_warn_once("installers.llamaindex.bind_failed", e)

    def teardown() -> None:
        for f in reversed(teardowns):
            try:
                f()
            except Exception as e:
                log_warn_once("installers.memory_taps.teardown_failed", e)

    return teardown


@contextmanager
def provider_memory_taps() -> Iterator[None]:
    """Context manager to install provider memory/retrieval taps.

    Installs best-effort patches for mem0, LangChain retrievers, and LlamaIndex; restores on exit.
    """
    undo = bind_memory_taps()
    try:
        yield None
    finally:
        try:
            undo()
        except Exception as e:
            log_warn_once("installers.provider_memory_taps.undo_failed", e)
