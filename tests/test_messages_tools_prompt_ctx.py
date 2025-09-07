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


def test_messages_aggregated_tools_and_prompt_ctx(tmp_path: Path) -> None:
    # Prepare messages chunks with available_tools and messages for prompt_ctx
    tools = [{"name": "search", "schema": {"type": "object"}}]
    meta = {
        "langgraph_node": "llm1",
        "ns": ["llm1"],
        "thread_id": "t1",
        "available_tools": tools,
        "messages": [{"role": "user", "content": "Hi"}],
    }
    msg1 = ({"content": "Hel"}, meta)
    msg2 = ({"content": "lo"}, meta)
    updates: list[tuple[str, Any]] = [
        ("messages", msg1),
        ("messages", msg2),
        ("updates", {"post": {"done": True}}),
    ]
    graph = _FakeGraph(updates)

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="agg-tools", framework="langgraph")
    rec = LangGraphRecorder(graph=graph, store=store, run=run)

    _ = rec.invoke({"x": 1}, config={})
    events = store.list_events(run.run_id)

    # Find aggregated LLM event
    llm = next(e for e in events if e.action_type is ActionType.LLM)
    assert llm.tools_digest is not None
    assert isinstance(llm.hashes.get("tools"), str)
    assert isinstance(llm.hashes.get("prompt_ctx"), str)
    assert llm.input_ref is not None
    # Blob decodes to {messages, tools}
    raw = store.get_blob(llm.input_ref)
    obj = orjson.loads(raw)
    assert isinstance(obj.get("messages"), list)
    assert isinstance(obj.get("tools"), list)
