from __future__ import annotations

from typing import Any

"""Parallel-ish LangGraph demo app.

Creates a small graph that fans out to two nodes, then joins. This demonstrates
multiple node completions and ordering anchoring via DECISION events.
"""


def make_graph_parallel() -> Any:
    try:  # pragma: no cover - optional dependency in CI
        from typing_extensions import TypedDict
    except Exception:  # pragma: no cover
        from typing import TypedDict  # type: ignore[no-redef]

    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("LangGraph is required for the parallel demo app") from exc

    class State(TypedDict):
        text: str
        a: int
        b: int

    def decide(state: State) -> dict[str, Any]:
        # Return a noop patch; routing is handled by graph edges
        return {}

    def branch_a(state: State) -> dict[str, Any]:
        return {"a": 1}

    def branch_b(state: State) -> dict[str, Any]:
        return {"b": 2}

    def join(state: State) -> dict[str, Any]:
        # combine results
        s = state.get("text", "")
        return {"text": f"{s}:{state.get('a', 0)}+{state.get('b', 0)}"}

    g = StateGraph(State)
    g.add_node("decide", decide)
    g.add_node("branch_a", branch_a)
    g.add_node("branch_b", branch_b)
    g.add_node("join", join)
    g.add_edge(START, "decide")
    # Fan-out (LangGraph will manage concurrency under the hood)
    g.add_edge("decide", "branch_a")
    g.add_edge("decide", "branch_b")
    # Join
    g.add_edge("branch_a", "join")
    g.add_edge("branch_b", "join")
    g.add_edge("join", END)
    saver = InMemorySaver()
    compiled = g.compile(checkpointer=saver)
    return compiled
