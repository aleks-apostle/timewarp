from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import orjson

from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _FakeGraph:
    def __init__(self, updates: list[tuple[str, Any]]) -> None:
        self._updates = updates

    def stream(
        self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any
    ) -> Iterable[Any]:
        yield from self._updates


def test_retrieval_detection_values_stream(tmp_path: Path) -> None:
    retrieval_payload = {
        "values": {
            "retrieval": {
                "query": "q1",
                "items": [{"id": "a", "text": "A", "score": 0.8}],
                "top_k": 1,
                "retriever": "vector",
                "query_id": "qid1",
            }
        }
    }
    updates: list[tuple[str, Any]] = [("values", retrieval_payload)]
    graph = _FakeGraph(updates)

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="retrieval", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        detect_retrieval=True,
    )

    _ = rec.invoke({"x": 1}, config={})
    events = store.list_events(run.run_id)

    # Expect a RETRIEVAL event present
    ret = next(e for e in events if e.action_type is ActionType.RETRIEVAL)
    assert ret.retriever == "vector"
    assert ret.top_k == 1
    assert ret.query_id == "qid1"
    assert ret.output_ref is not None
    raw = store.get_blob(ret.output_ref)
    obj = orjson.loads(raw)
    assert obj.get("query") == "q1"
    assert isinstance(obj.get("items"), list) and len(obj.get("items")) == 1
    pol = obj.get("policy")
    assert isinstance(pol, dict) and pol.get("top_k") == 1
