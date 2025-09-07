from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

from timewarp.bindings import begin_recording_session, bind_memory_taps
from timewarp.events import ActionType, Run
from timewarp.langgraph.taps import flush_provider_taps
from timewarp.store import LocalStore


class _NodeWithScore:
    def __init__(self, text: str, score: float) -> None:
        self.text = text
        self.score = score

    def model_dump(self) -> dict[str, Any]:
        return {"text": self.text, "score": self.score}


class _LIRetrieverBase:
    k: int = 2

    def retrieve(self, query: str, **_: Any) -> list[Any]:  # original impl
        return [_NodeWithScore("A", 0.9), _NodeWithScore("B", 0.8)]


@pytest.fixture(autouse=True)
def _inject_fake_llamaindex(monkeypatch: pytest.MonkeyPatch) -> None:
    m = ModuleType("llama_index")
    m.BaseRetriever = _LIRetrieverBase
    monkeypatch.setitem(__import__("sys").modules, "llama_index", m)


def test_llamaindex_retriever_tap_emits_retrieval(tmp_path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="li", framework="langgraph")
    store.create_run(run)

    teardown = bind_memory_taps()
    try:
        end = begin_recording_session(run.run_id)
        # After bind_memory_taps, BaseRetriever.retrieve is wrapped but still calls orig
        from llama_index import BaseRetriever  # type: ignore

        r = BaseRetriever()
        _ = r.retrieve("q", top_k=4)
        evs, _ = flush_provider_taps(
            store=store,
            run_id=run.run_id,
            step=0,
            actor="graph",
            namespace_label=None,
            thread_id=None,
            adapter_version="test",
            privacy_marks=None,
            pruner=None,
        )
        for e in evs:
            store.append_event(e)
        end()
    finally:
        teardown()

    events = store.list_events(run.run_id)
    rets = [e for e in events if e.action_type is ActionType.RETRIEVAL]
    assert rets, "expected retrieval event"
    ret = rets[0]
    assert ret.mem_provider == "LlamaIndex"
    assert ret.top_k == 4 or ret.top_k == 2  # prefer kwarg; else default from attr
    # results digest present
    assert "results" in (ret.hashes or {})
