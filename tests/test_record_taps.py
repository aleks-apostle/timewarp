from __future__ import annotations

from pathlib import Path

import pytest

from timewarp.bindings import begin_recording_session, bind_langgraph_record
from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore

try:
    from typing import TypedDict

    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    DEPS = True
except Exception:  # pragma: no cover - optional dependency
    DEPS = False


@pytest.mark.skipif(not DEPS, reason="langgraph/langchain-core not installed")
def test_record_taps_add_prompt_hash(tmp_path: Path) -> None:
    class State(TypedDict):
        text: str

    llm = FakeListChatModel(responses=["ignored"])  # model output irrelevant; taps target prompt

    def compose(state: State) -> dict[str, object]:
        _ = llm.invoke("hello taps")
        return {"text": state["text"] + "!"}

    g = StateGraph(State)
    g.add_node("compose", compose)
    g.add_edge(START, "compose")
    g.add_edge("compose", END)
    saver = InMemorySaver()
    compiled = g.compile(checkpointer=saver)

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="taps", framework="langgraph")

    teardown = bind_langgraph_record()
    try:
        # Ensure a recording session is active so staged hashes are scoped
        end_session = begin_recording_session(run.run_id)
        rec = LangGraphRecorder(
            graph=compiled,
            store=store,
            run=run,
            snapshot_every=0,
            stream_modes=("messages",),  # surface LLM via messages stream
            stream_subgraphs=False,
        )
        _ = rec.invoke({"text": "hi"}, config={"configurable": {"thread_id": "t1"}})
    finally:
        try:
            end_session()
        except Exception:
            pass
        teardown()

    events = store.list_events(run.run_id)
    # Expect an LLM event with a prompt hash present (from taps or stream metadata)
    llm_ev = next(e for e in events if e.action_type is ActionType.LLM)
    assert "prompt" in llm_ev.hashes and isinstance(llm_ev.hashes["prompt"], str)
