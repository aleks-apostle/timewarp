from __future__ import annotations

from typing import Any, Protocol

from ..events import ActionType


class ToolClassifier(Protocol):
    def __call__(self, tool: Any) -> dict[str, str] | None:
        """Return MCP/tool metadata if tool is MCP, else None.

        Expected keys when MCP: {"tool_kind": "MCP", "tool_name": str,
        "mcp_server": str, "mcp_transport": str}
        """


def default_tool_classifier() -> ToolClassifier:
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


def infer_action_type(update: Any) -> ActionType:
    """Heuristic ActionType from update payload.

    - TOOL when tool/tool_name present
    - LLM when messages-like metadata present
    - otherwise SYS
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
        if isinstance(obs, dict):
            if "tool" in obs or "tool_name" in obs or obs.get("tool_kind"):
                return ActionType.TOOL
            if "messages" in obs or "llm_input_messages" in obs:
                return ActionType.LLM
    except Exception:
        pass
    return ActionType.SYS


def classify_tool_from_update(
    update: Any, tool_classifier: ToolClassifier | None
) -> dict[str, str] | None:
    """Return normalized MCP metadata from an update if available.

    Uses a user-provided classifier when given, then falls back to best-effort
    heuristics over the update's structure and nested metadata.
    """
    obs = update
    try:
        if isinstance(update, dict) and len(update) == 1:
            ((_, inner),) = update.items()
            if isinstance(inner, dict):
                obs = inner
    except Exception:
        obs = update

    if tool_classifier and hasattr(obs, "get"):
        tool_obj = obs.get("tool") if isinstance(obs, dict) else None
        if tool_obj is not None:
            meta = tool_classifier(tool_obj)
            if meta:
                out: dict[str, str] = {
                    "tool_kind": meta.get("tool_kind", "MCP") or "MCP",
                    "tool_name": meta.get("tool_name", "unknown") or "unknown",
                }
                if meta.get("mcp_server") is not None:
                    out["mcp_server"] = str(meta["mcp_server"])  # ensure str
                if meta.get("mcp_transport") is not None:
                    out["mcp_transport"] = str(meta["mcp_transport"])  # ensure str
                return out

    if isinstance(obs, dict):
        name = obs.get("tool_name") or obs.get("name")
        kind = obs.get("tool_kind")
        if name and (kind == "MCP" or obs.get("mcp_server") or obs.get("mcp_transport")):
            out2: dict[str, str] = {
                "tool_kind": str(kind or "MCP"),
                "tool_name": str(name),
            }
            if obs.get("mcp_server"):
                out2["mcp_server"] = str(obs.get("mcp_server"))
            if obs.get("mcp_transport"):
                out2["mcp_transport"] = str(obs.get("mcp_transport"))
            return out2
        meta = obs.get("metadata")
        if isinstance(meta, dict):
            name2 = meta.get("tool_name") or meta.get("name")
            kind2 = meta.get("tool_kind")
            if name2 and (kind2 == "MCP" or meta.get("mcp_server") or meta.get("mcp_transport")):
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
