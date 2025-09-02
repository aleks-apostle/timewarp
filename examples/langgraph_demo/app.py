from __future__ import annotations

from typing import Any

"""Minimal LangGraph demo app factory for Timewarp CLI.

Provides `make_graph()` returning a compiled graph with a single node that
optionally calls a LangChain Fake chat model if available.
"""


def make_graph() -> Any:
    try:
        from typing_extensions import TypedDict
    except Exception:  # pragma: no cover - test envs
        from typing import TypedDict  # type: ignore[no-redef]

    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("LangGraph is required for the demo app") from exc

    class State(TypedDict):
        text: str

    # Optional Fake chat model
    try:  # pragma: no cover - optional path
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        fake_llm = FakeListChatModel(responses=["hello from llm"])

        def compose(state: State) -> dict[str, Any]:
            _ = fake_llm.invoke("Say hello")  # content ignored; proves model call exists
            return {"text": state["text"] + "!"}

    except Exception:  # pragma: no cover - fallback path

        def compose(state: State) -> dict[str, Any]:
            return {"text": state["text"] + "!"}

    g = StateGraph(State)
    g.add_node("compose", compose)
    g.add_edge(START, "compose")
    g.add_edge("compose", END)
    saver = InMemorySaver()
    compiled = g.compile(checkpointer=saver)
    return compiled
