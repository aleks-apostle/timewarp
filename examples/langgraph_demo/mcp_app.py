from __future__ import annotations

from typing import Any

"""LangGraph MCP demo app (best-effort; optional deps).

Provides `make_graph_mcp()` returning a compiled graph that calls an MCP tool
via LangChain MCP adapters when available. If required dependencies are not
installed, calling the factory raises a RuntimeError.
"""


def make_graph_mcp() -> Any:
    try:
        from typing_extensions import TypedDict
    except Exception:  # pragma: no cover - fallback for older Python
        from typing import TypedDict  # type: ignore[no-redef]

    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("LangGraph is required for the MCP demo app") from exc

    # Attempt to import MCP tool adapter. We keep this best-effort and avoid hard deps.
    try:
        from langchain_mcp_adapters.tools import MCPTool  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("langchain-mcp-adapters is required for the MCP demo app") from exc

    class State(TypedDict):
        text: str
        result: str

    # Construct a basic MCP tool (echo-like). The exact constructor signature may vary
    # across versions; users can adapt this example to their environment.
    try:
        # Some adapter versions expose a simple MCPTool taking name and server details.
        mcp_tool = MCPTool(name="echo", server="stdio://echo", transport="stdio")  # type: ignore[call-arg]
    except Exception:  # pragma: no cover - demo variability
        # Fallback to a dummy callable with MCP-like attributes for demonstration
        class DummyTool:
            tool_name = "echo"
            mcp_server = "stdio://echo"
            mcp_transport = "stdio"

            def __call__(self, *args: Any, **kwargs: Any) -> str:
                return "ok"

        mcp_tool = DummyTool()

    def use_mcp(state: State) -> dict[str, Any]:
        try:
            res = mcp_tool({"text": state["text"]})  # type: ignore[misc]
        except Exception:
            # Best-effort: if adapter signature differs, just call without args
            try:
                res = mcp_tool()  # type: ignore[call-arg]
            except Exception:
                res = "ok"
        return {"result": str(res)}

    g = StateGraph(State)
    g.add_node("use_mcp", use_mcp)
    g.add_edge(START, "use_mcp")
    g.add_edge("use_mcp", END)
    saver = InMemorySaver()
    compiled = g.compile(checkpointer=saver)
    return compiled
