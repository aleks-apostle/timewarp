from __future__ import annotations

from typing import Any

from ..utils.hashing import hash_prompt_ctx as _hash_prompt_ctx
from ..utils.hashing import hash_tools_list as _hash_tools_list


def hash_tools_list(tools: list[Any]) -> str:
    return _hash_tools_list(tools)


def hash_prompt_ctx(*, messages: Any, tools: list[Any]) -> str:
    return _hash_prompt_ctx(messages=messages, tools=tools)


def extract_tools_from_update(update: Any) -> list[Any] | None:
    """Best-effort discovery of tools list within a LangGraph update payload."""
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


def extract_tool_args(update: Any) -> dict[str, Any] | None:
    """Normalize tool-call arguments from an update into {"args": [...], "kwargs": {...}}."""
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
