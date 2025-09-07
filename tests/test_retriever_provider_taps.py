from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from timewarp.bindings import begin_recording_session, bind_memory_taps
from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore

try:  # optional dependencies
    from typing import TypedDict

    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    DEPS = True
except Exception:  # pragma: no cover - skip when deps unavailable
    DEPS = False


@pytest.mark.skipif(not DEPS, reason="langgraph/langchain-core not installed")
def test_langchain_retriever_tap_emits_retrieval(tmp_path: Path) -> None:
    class State(TypedDict):
        text: str

    class MyRetriever(BaseRetriever):
        # Provide a top-k hint via a field
        k: int = 2

        def _get_relevant_documents(self, query: str, *, run_manager: Any | None = None, **_: Any):
            return [Document(page_content="A"), Document(page_content="B")]

    def compose(state: State) -> dict[str, object]:
        r = MyRetriever()
        docs = r.get_relevant_documents("hello")
        return {"text": state["text"] + str(len(docs))}

    g = StateGraph(State)
    g.add_node("compose", compose)
    g.add_edge(START, "compose")
    g.add_edge("compose", END)
    saver = InMemorySaver()
    compiled = g.compile(checkpointer=saver)

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="retriever-taps", framework="langgraph")

    teardown_mem = bind_memory_taps()
    try:
        end_session = begin_recording_session(run.run_id)
        rec = LangGraphRecorder(
            graph=compiled,
            store=store,
            run=run,
            snapshot_every=0,
            stream_modes=("updates",),
            stream_subgraphs=False,
        )
        _ = rec.invoke({"text": ""}, config={"configurable": {"thread_id": "t1"}})
    finally:
        try:
            end_session()
        except Exception:
            pass
        teardown_mem()

    events = store.list_events(run.run_id)
    # Find a RETRIEVAL event emitted from the provider tap
    rets = [e for e in events if e.action_type is ActionType.RETRIEVAL]
    assert len(rets) >= 1
    ret = rets[0]
    # Anchor id should be present for diff alignment
    assert isinstance(ret.labels.get("anchor_id"), str)
    # Policy/top_k propagated
    assert ret.top_k == 2
    # Provider tagged as LangChainRetriever:MyRetriever
    assert (
        isinstance(ret.mem_provider, str) and "LangChainRetriever:MyRetriever" in ret.mem_provider
    )
    # Hashes include results digest
    assert "results" in ret.hashes
