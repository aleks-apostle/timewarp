from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _FakeGraph:
    """Minimal mock of a compiled LangGraph supporting .stream().

    Yields a sequence of ('messages', (msg_chunk, metadata)) tuples and a
    trailing ('updates', {...}) entry to force a flush boundary.
    """

    def __init__(self, updates: list[tuple[str, Any]]) -> None:
        self._updates = updates

    def stream(
        self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any
    ) -> Iterable[Any]:
        yield from self._updates


def test_messages_aggregation_single_event(tmp_path: Path) -> None:
    # Prepare a fake stream with three messages chunks, same actor namespace
    msg1 = ({"content": "hel"}, {"langgraph_node": "llm1", "ns": ["llm1"], "thread_id": "t1"})
    msg2 = ({"content": "lo "}, {"langgraph_node": "llm1", "ns": ["llm1"], "thread_id": "t1"})
    msg3 = ({"content": "world"}, {"langgraph_node": "llm1", "ns": ["llm1"], "thread_id": "t1"})
    updates: list[tuple[str, Any]] = [
        ("messages", msg1),
        ("messages", msg2),
        ("messages", msg3),
        ("updates", {"post": {"done": True}}),  # boundary to force flush
    ]
    graph = _FakeGraph(updates)

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="agg", framework="langgraph")
    rec = LangGraphRecorder(graph=graph, store=store, run=run)

    _ = rec.invoke({"x": 1}, config={})
    events = store.list_events(run.run_id)

    # Expect: initial SYS + single LLM aggregated event + trailing updates SYS event
    assert len(events) == 3
    assert events[0].action_type is ActionType.SYS
    llm_ev = events[1]
    assert llm_ev.action_type is ActionType.LLM
    assert llm_ev.labels.get("stream_mode") == "messages"
    assert llm_ev.labels.get("namespace") == "llm1"
    assert llm_ev.labels.get("thread_id") == "t1"
    assert llm_ev.model_meta and llm_ev.model_meta.get("chunks_count") == 3
    # Check aggregated content
    raw = store.get_blob(llm_ev.output_ref) if llm_ev.output_ref else None
    assert raw is not None
    import orjson as _orjson

    obj = _orjson.loads(raw)
    assert obj.get("message", {}).get("content") == "hello world"
    # Optional chunks_ref present and decodes to a chunks list of length 3
    cref = obj.get("chunks_ref")
    assert isinstance(cref, dict)
    from timewarp.events import BlobRef as _BlobRef

    bref = _BlobRef.model_validate(cref)
    chunks_raw = store.get_blob(bref)
    chunks_obj = _orjson.loads(chunks_raw)
    assert isinstance(chunks_obj.get("chunks"), list)
    assert len(chunks_obj["chunks"]) == 3


def test_replay_state_prefers_values_stream(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="values", framework="langgraph")
    store.create_run(run)

    # Initial SYS input event with no state snapshot
    import orjson as _orjson

    from timewarp.events import BlobKind, Event

    bref = store.put_blob(run.run_id, 0, BlobKind.INPUT, _orjson.dumps({"a": 1}))
    store.append_event(
        Event(
            run_id=run.run_id,
            step=0,
            action_type=ActionType.SYS,
            actor="graph",
            input_ref=bref,
            hashes={"input": bref.sha256_hex},
        )
    )
    # Later values-stream events
    v1 = store.put_blob(run.run_id, 1, BlobKind.OUTPUT, _orjson.dumps({"state": {"x": 1}}))
    store.append_event(
        Event(
            run_id=run.run_id,
            step=1,
            action_type=ActionType.SYS,
            actor="graph",
            output_ref=v1,
            hashes={"output": v1.sha256_hex},
            labels={"stream_mode": "values"},
        )
    )
    v2 = store.put_blob(run.run_id, 2, BlobKind.OUTPUT, _orjson.dumps({"state": {"x": 2}}))
    store.append_event(
        Event(
            run_id=run.run_id,
            step=2,
            action_type=ActionType.SYS,
            actor="graph",
            output_ref=v2,
            hashes={"output": v2.sha256_hex},
            labels={"stream_mode": "values"},
        )
    )

    from timewarp.replay import Replay

    rep = Replay(store=store, run_id=run.run_id)
    rep.goto(3)
    state = rep.inspect_state()
    assert isinstance(state, dict)
    assert state.get("state", {}).get("x") == 2
