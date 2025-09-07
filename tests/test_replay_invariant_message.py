from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import pytest

from timewarp.events import Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.replay import AdapterInvariant, LangGraphReplayer
from timewarp.store import LocalStore

try:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langgraph.graph import END, START, StateGraph

    LG_AVAILABLE = True
except Exception:  # pragma: no cover - optional deps
    LG_AVAILABLE = False


@pytest.mark.skipif(not LG_AVAILABLE, reason="langgraph/langchain-core not installed")
def test_replay_requires_wrappers_message(tmp_path: Path) -> None:
    class State(TypedDict):
        text: str

    # LLM to produce an LLM event in messages stream
    llm = FakeListChatModel(responses=["hello world"])  # deterministic

    def node(state: State) -> dict[str, Any]:
        _ = llm.invoke("Say hi")
        return {"text": state["text"] + "!"}

    g = StateGraph(State)
    g.add_node("call_model", node)
    g.add_edge(START, "call_model")
    g.add_edge("call_model", END)
    compiled = g.compile()

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="msg", framework="langgraph")
    rec = LangGraphRecorder(
        graph=compiled,
        store=store,
        run=run,
        stream_modes=("messages",),
        stream_subgraphs=False,
    )
    _ = rec.invoke({"text": "ok"}, config={})

    # Attempt replay without wrappers: should raise with actionable guidance
    replayer = LangGraphReplayer(graph=compiled, store=store)
    with pytest.raises(AdapterInvariant) as ei:
        _ = replayer.resume(run.run_id, from_step=None, thread_id=None, install_wrappers=None)
    msg = str(ei.value)
    assert "Playback wrappers required" in msg
    assert "bind_langgraph_playback" in msg
    assert "langchain-core" in msg or "optional deps" in msg
    assert "CLI `resume`" in msg
