from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore

try:
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LANGGRAPH_AVAILABLE = False


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="langgraph not installed")
def test_langgraph_recorder_single_execution(tmp_path: Path) -> None:
    class State(TypedDict):
        val: int

    counter = {"n": 0}

    def inc_node(state: State) -> State:
        counter["n"] += 1
        return {"val": state["val"] + 1}

    # Build a minimal graph: START -> inc_node -> END
    graph = StateGraph(State)
    graph.add_node("inc", inc_node)
    graph.add_edge(START, "inc")
    graph.add_edge("inc", END)
    compiled = graph.compile()

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="n", framework="langgraph")

    rec = LangGraphRecorder(
        graph=compiled,
        store=store,
        run=run,
        snapshot_every=20,
        stream_modes=("values",),  # Ensure we get full state values in stream
        stream_subgraphs=False,
        require_thread_id=False,
    )

    result = rec.invoke({"val": 1}, config={})

    # Node executed exactly once, no post-stream re-invocation
    assert counter["n"] == 1

    # Result comes from final state values
    assert isinstance(result, dict)
    assert result["val"] == 2

    events = store.list_events(run.run_id)
    # initial SYS input + at least one streamed step
    assert len(events) >= 2
    assert events[0].action_type is ActionType.SYS and events[0].input_ref is not None
    # Ensure at least one streamed output event exists
    assert any(
        e.output_ref is not None and e.action_type in (ActionType.SYS, ActionType.LLM)
        for e in events[1:]
    )


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="langgraph not installed")
def test_langgraph_recorder_multi_mode_with_subgraphs(tmp_path: Path) -> None:
    class SubState(TypedDict):
        foo: str
        bar: str

    class ParentState(TypedDict):
        foo: str

    # Subgraph
    def subgraph_node_1(state: SubState) -> dict[str, Any]:
        return {"bar": "bar"}

    def subgraph_node_2(state: SubState) -> dict[str, Any]:
        return {"foo": state["foo"] + state["bar"]}

    sub_builder = StateGraph(SubState)
    sub_builder.add_node("subgraph_node_1", subgraph_node_1)
    sub_builder.add_node("subgraph_node_2", subgraph_node_2)
    sub_builder.add_edge(START, "subgraph_node_1")
    sub_builder.add_edge("subgraph_node_1", "subgraph_node_2")
    subgraph = sub_builder.compile()

    # Parent graph
    def node_1(state: ParentState) -> dict[str, Any]:
        return {"foo": "hi! " + state["foo"]}

    parent = StateGraph(ParentState)
    parent.add_node("node_1", node_1)
    parent.add_node("node_2", subgraph)
    parent.add_edge(START, "node_1")
    parent.add_edge("node_1", "node_2")
    compiled = parent.compile()

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="multi", framework="langgraph")

    rec = LangGraphRecorder(
        graph=compiled,
        store=store,
        run=run,
        snapshot_every=50,
        stream_modes=("updates", "values"),  # list mode → (mode, chunk)
        stream_subgraphs=True,
    )

    result = rec.invoke({"foo": "foo"}, config={})
    assert isinstance(result, dict)
    # from subgraph: hi! foo + bar becomes hi! foobar
    assert result.get("foo") == "hi! foobar"

    events = store.list_events(run.run_id)
    # Initial SYS + several streamed events
    assert len(events) >= 3
    # Expect at least one event with a namespace label and stream_mode recorded
    assert any(("namespace" in e.labels) or ("stream_mode" in e.labels) for e in events[1:])


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="langgraph not installed")
def test_langgraph_recorder_messages_mode(tmp_path: Path) -> None:
    # Build a graph with a node that calls a (fake) LLM via LangChain core
    try:
        from langchain_core.language_models.fake_chat_models import (
            FakeListChatModel,
        )
    except Exception:  # pragma: no cover - optional dependency issues
        pytest.skip("langchain-core not available")

    class State(TypedDict):
        text: str

    llm = FakeListChatModel(responses=["hello world"])  # deterministic fake chat model

    def call_model(state: State) -> dict[str, Any]:
        # Even with invoke(), messages stream should emit events
        _ = llm.invoke("Say hi")
        return {"text": state["text"] + "!"}

    graph = StateGraph(State)
    graph.add_node("call_model", call_model)
    graph.add_edge(START, "call_model")
    compiled = graph.compile()

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="msgs", framework="langgraph")

    rec = LangGraphRecorder(
        graph=compiled,
        store=store,
        run=run,
        snapshot_every=50,
        stream_modes=("messages",),  # single-mode messages → (message_chunk, metadata)
        stream_subgraphs=False,
    )

    _ = rec.invoke({"text": "ok"}, config={})

    events = store.list_events(run.run_id)
    # We expect initial SYS input + at least one LLM/messages event
    assert len(events) >= 2
    assert any(
        e.labels.get("stream_mode") == "messages" and e.action_type is ActionType.LLM
        for e in events[1:]
    )
