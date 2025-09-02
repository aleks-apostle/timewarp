from __future__ import annotations

from typing import Any

"""LangGraph demo app for Timewarp freeze-time example.

Provides `make_graph_time()` returning a compiled graph that records the current
Timewarp time (`timewarp.determinism.now()`) into the state. When replayed with
`--freeze-time`, the timestamp remains identical to the recorded run.
"""


def make_graph_time() -> Any:
    try:  # pragma: no cover - test envs fallback
        from typing_extensions import TypedDict
    except Exception:  # pragma: no cover
        from typing import TypedDict  # type: ignore[no-redef]

    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("LangGraph is required for the demo app") from exc

    from timewarp.determinism import now as tw_now

    class State(TypedDict):
        text: str
        now_iso: str

    def annotate_time(state: State) -> dict[str, Any]:
        current = tw_now()
        return {"now_iso": current.isoformat()}

    def compose(state: State) -> dict[str, Any]:
        # trivial mutation to show multiple nodes still keep frozen time semantics
        return {"text": state["text"] + "!"}

    g = StateGraph(State)
    g.add_node("annotate_time", annotate_time)
    g.add_node("compose", compose)
    g.add_edge(START, "annotate_time")
    g.add_edge("annotate_time", "compose")
    g.add_edge("compose", END)
    saver = InMemorySaver()
    compiled = g.compile(checkpointer=saver)
    return compiled
