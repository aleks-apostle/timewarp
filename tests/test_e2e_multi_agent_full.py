from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import orjson as _orjson
import pytest

from timewarp import messages_pruner, wrap
from timewarp.bindings import stage_memory_tap
from timewarp.diff import bisect_divergence, first_divergence
from timewarp.events import ActionType
from timewarp.exporters.langsmith import serialize_run
from timewarp.replay import Replay
from timewarp.store import LocalStore

# Optional dependency: langgraph
try:  # pragma: no cover - optional dep
    from examples.langgraph_demo.multi_agent_full import make_graph_multi

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep environment
    LANGGRAPH_AVAILABLE = False


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="langgraph not installed")
def test_e2e_multi_agent_full_demo_exercises_features(tmp_path: Path) -> None:
    # Set up an isolated store
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)

    # Build the demo graph (sync variant is sufficient for end-to-end coverage)
    graph = make_graph_multi(include_async=False)

    # Stage memory provider taps to exercise MEMORY and RETRIEVAL events end-to-end
    stage_memory_tap(
        {
            "kind": "MEMORY",
            "mem_provider": "Mem0",
            "mem_op": "PUT",
            "key": "long.session",
            "value": {"k": "v", "n": 1},
            "mem_scope": "long",
            "mem_space": "demo",
        }
    )
    stage_memory_tap(
        {
            "kind": "RETRIEVAL",
            "mem_provider": "Mem0",
            "query": "demo query",
            "items": [
                {"id": "a", "text": "A", "score": 0.9},
                {"id": "b", "text": "B", "score": 0.7},
            ],
            "policy": {"retriever": "vector", "top_k": 2},
            "query_id": "qid-demo",
        }
    )

    # Record the run through the facade to enable record-time taps and redaction
    rec = wrap(
        graph,
        project="demo",
        name="full-e2e",
        store=store,
        snapshot_every=5,
        snapshot_on=("terminal", "decision"),
        stream_modes=("updates", "messages", "values"),
        state_pruner=messages_pruner(max_len=500, max_items=50),
        enable_record_taps=True,
        enable_memory_taps=True,
        event_batch_size=20,
        privacy_marks={"kwargs.ssn": "mask4"},
    )
    cfg: dict[str, Any] = {"configurable": {"thread_id": "t-demo"}}
    result = rec.invoke({"messages": [{"role": "user", "content": "do work"}]}, config=cfg)
    run_id = cast(Any, rec.last_run_id)

    # Sanity: result contains a report from compose node
    assert isinstance(result, dict)
    # Best-effort: tolerate different shapes; prefer values if present
    res_vals = cast(
        dict[str, Any] | None,
        result.get("values") if isinstance(result.get("values"), dict) else result,
    )
    assert isinstance(res_vals, dict)
    assert "report" in res_vals

    # Load events and verify broad feature coverage
    events = store.list_events(run_id)
    assert len(events) > 0
    steps = [e.step for e in events]
    assert steps == sorted(steps)  # serialized completion order

    kinds = {e.action_type for e in events}
    # Core kinds from the demo + staged memory taps
    assert ActionType.SYS in kinds
    assert ActionType.LLM in kinds
    assert ActionType.TOOL in kinds
    assert ActionType.DECISION in kinds
    assert ActionType.HITL in kinds
    assert ActionType.SNAPSHOT in kinds
    assert ActionType.MEMORY in kinds
    assert ActionType.RETRIEVAL in kinds

    # Initial SYS event includes input blob hash and RNG snapshot
    sys0 = events[0]
    assert sys0.action_type is ActionType.SYS
    assert sys0.input_ref is not None and sys0.hashes.get("input") is not None
    assert isinstance(sys0.rng_state, bytes | bytearray)

    # Verify at least one LLM event has a prompt hash and metadata
    llm_ev = next(e for e in events if e.action_type is ActionType.LLM)
    assert llm_ev.hashes.get("prompt") is not None or llm_ev.hashes.get("prompt_ctx") is not None
    assert isinstance(llm_ev.model_meta, dict) and llm_ev.model_meta.get("framework") == "langgraph"

    # Verify TOOL event classification + args hashing + privacy redaction persisted to blob
    tool_ev = next(e for e in events if e.action_type is ActionType.TOOL)
    assert tool_ev.tool_kind == "MCP"
    assert tool_ev.tool_name == "calc"
    assert tool_ev.mcp_server is not None and tool_ev.mcp_transport is not None
    assert isinstance(tool_ev.hashes.get("args"), str)
    assert tool_ev.output_ref is not None and tool_ev.output_ref.compression == "zstd"
    tool_payload = _orjson.loads(store.get_blob(tool_ev.output_ref))
    # kwargs.ssn should be masked by privacy_marks
    if isinstance(tool_payload, dict):
        # Tool payload recorded either directly or nested depending on stream normalization
        kp = (
            tool_payload.get("kwargs")
            if "kwargs" in tool_payload
            else (
                tool_payload.get("tool_args", {}).get("kwargs")
                if isinstance(tool_payload.get("tool_args"), dict)
                else None
            )
        )
        if isinstance(kp, dict) and "ssn" in kp:
            assert (
                isinstance(kp["ssn"], str)
                and kp["ssn"].endswith("6789")
                and kp["ssn"].startswith("***")
            )

    # Verify DECISION event and that snapshots exist (terminal + decision)
    dec = next(e for e in events if e.action_type is ActionType.DECISION)
    assert isinstance(dec.labels.get("decision"), str)
    snaps = [e for e in events if e.action_type is ActionType.SNAPSHOT]
    assert len(snaps) >= 1
    # Terminal snapshot should carry thread_id and maybe checkpoint label
    assert any("thread_id" in (s.labels or {}) for s in snaps)

    # Staged MEMORY and RETRIEVAL events present and well-formed
    mem_ev = next(e for e in events if e.action_type is ActionType.MEMORY)
    ret_ev = next(e for e in events if e.action_type is ActionType.RETRIEVAL)
    assert mem_ev.output_ref is not None and ret_ev.output_ref is not None
    ret_payload = _orjson.loads(store.get_blob(ret_ev.output_ref))
    assert ret_ev.retriever == "vector" and ret_ev.top_k == 2 and ret_ev.query_id == "qid-demo"
    assert (
        isinstance(ret_payload.get("items"), list)
        and len(cast(list[Any], ret_payload.get("items"))) == 2
    )

    # Exporter serializes run + events with optional inline blobs
    export_payload = serialize_run(store, run_id, include_blobs=True)
    assert "run" in export_payload and "events" in export_payload
    assert isinstance(export_payload["events"], list) and len(
        cast(list[Any], export_payload["events"])
    ) == len(events)
    # SYS event inlined input payload mirrors the recording input
    ev0 = cast(dict[str, Any], export_payload["events"][0])
    if ev0.get("input_ref") and ev0.get("input_payload"):
        ip = ev0.get("input_payload")
        assert isinstance(ip, dict) and "messages" in ip

    # Deterministic resume using recorded outputs via Replay.resume
    factory = "examples.langgraph_demo.multi_agent_full:make_graph_multi"
    sess = Replay.resume(
        store, app_factory=factory, run_id=run_id, thread_id="t-demo", freeze_time=True
    )
    assert sess.result is None or isinstance(sess.result, dict)

    # Replay overlay injection on a values event modifies reconstructed state
    # Locate a 'values' event carrying a report
    tgt_values = None
    for e in events:
        if e.labels.get("stream_mode") == "values" and e.output_ref is not None:
            payload = _orjson.loads(store.get_blob(e.output_ref))
            # Accept either direct values or a singleton node dict containing values-like content
            obj: dict[str, Any] | None
            if (
                isinstance(payload, dict)
                and "values" in payload
                and isinstance(payload["values"], dict)
            ):
                obj = payload["values"]
            elif isinstance(payload, dict) and len(payload) == 1:
                ((_, only),) = tuple(payload.items())
                obj = only if isinstance(only, dict) else None
            else:
                obj = None
            if isinstance(obj, dict) and "report" in obj:
                tgt_values = (e.step, payload)
                break
    assert tgt_values is not None
    step_to_inject, orig_payload = tgt_values

    # Build an overlay that tweaks the report text
    new_payload = _orjson.loads(_orjson.dumps(orig_payload))  # deep copy via JSON
    try:
        if isinstance(new_payload, dict):
            if "values" in new_payload and isinstance(new_payload["values"], dict):
                new_payload["values"]["report"] = "[replayed override]"
            elif len(new_payload) == 1:
                ((only_k, only_v),) = tuple(new_payload.items())
                if isinstance(only_v, dict):
                    only_v["report"] = "[replayed override]"
                    new_payload[only_k] = only_v
    except Exception:
        # If structure is unexpected, fall back to a minimal patch
        new_payload = {"values": {"report": "[replayed override]"}}

    rp = Replay(store=store, run_id=run_id)
    rp.goto(step_to_inject + 1)
    # Baseline: report reflects original
    base_state = cast(dict[str, Any] | None, rp.inspect_state())
    assert isinstance(base_state, dict)
    base_report = cast(str, base_state.get("report"))
    rp.inject(step_to_inject, new_payload)
    rp.goto(step_to_inject + 1)
    mod_state = cast(dict[str, Any] | None, rp.inspect_state())
    assert isinstance(mod_state, dict)
    assert (
        mod_state.get("report") == "[replayed override]" or mod_state.get("report") != base_report
    )

    # Fork a what-if by overriding the first LLM step and record the branch
    from timewarp.events import Run as _Run
    from timewarp.langgraph import LangGraphRecorder
    from timewarp.replay import LangGraphReplayer

    llm_step = next(e.step for e in events if e.action_type is ActionType.LLM)
    graph2 = make_graph_multi(include_async=False)
    replayer = LangGraphReplayer(graph=graph2, store=store)

    # Bind playback wrappers using the installers so the graph runs side-effect free
    teardowns: list[Any] = []

    def _installer(llm: Any, tool: Any, _memory: Any) -> None:
        from timewarp.bindings import bind_langgraph_playback

        td = bind_langgraph_playback(graph2, llm, tool, _memory)
        teardowns.append(td)

    alt_output = {"message": {"content": "[what-if override]"}}
    new_run_id = replayer.fork_with_injection(
        run_id,
        llm_step,
        alt_output,
        thread_id="t-demo",
        install_wrappers=_installer,
        freeze_time=True,
    )

    try:
        # Re-record the fork under the provided run id with same input
        orig_input: dict[str, Any] | None = None
        for e in events:
            if e.input_ref is not None:
                orig_input = cast(dict[str, Any], _orjson.loads(store.get_blob(e.input_ref)))
                break
        assert isinstance(orig_input, dict)
        fork_run = _Run(
            run_id=new_run_id,
            project="demo",
            name="full-e2e-fork",
            framework="langgraph",
            labels={"branch_of": str(run_id), "override_step": str(llm_step)},
        )
        rec2 = LangGraphRecorder(
            graph=graph2,
            store=store,
            run=fork_run,
            snapshot_every=5,
            snapshot_on={"terminal", "decision"},
            stream_modes=("updates", "messages", "values"),
            stream_subgraphs=True,
        )
        _ = rec2.invoke(orig_input, config=cfg)

        # Diff analysis between baseline and fork: expect first divergence at the injected step
        fd = first_divergence(store, run_id, new_run_id, window=5)
        assert fd is not None and fd.reason == "output hash mismatch" and fd.step_a == llm_step
        bi = bisect_divergence(store, run_id, new_run_id, window=5)
        assert bi is not None and cast(dict[str, Any], bi).get("cause") in {
            "output hash mismatch",
            "anchor mismatch",
        }
    finally:
        for td in teardowns:
            try:
                td()
            except Exception:
                pass


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="langgraph not installed")
def test_e2e_multi_agent_full_demo_async(tmp_path: Path) -> None:
    # Build async-capable graph variant
    graph = make_graph_multi(include_async=True)
    if not (hasattr(graph, "astream") and callable(graph.astream)):
        pytest.skip("graph does not expose .astream")

    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)

    # Stage a lightweight set of provider taps to exercise async flush path
    stage_memory_tap(
        {
            "kind": "RETRIEVAL",
            "mem_provider": "Mem0",
            "query": "q-async",
            "items": [{"id": "x", "text": "X"}],
            "policy": {"retriever": "vector", "top_k": 1},
            "query_id": "qid-async",
        }
    )

    rec = wrap(
        graph,
        project="demo",
        name="full-e2e-async",
        store=store,
        snapshot_every=5,
        snapshot_on=("terminal", "decision"),
        stream_modes=("updates", "messages", "values"),
        state_pruner=messages_pruner(max_len=500, max_items=50),
        enable_record_taps=True,
        enable_memory_taps=True,
        event_batch_size=20,
    )

    cfg: dict[str, Any] = {"configurable": {"thread_id": "t-demo-async"}}

    # Run async recording
    import asyncio

    async def _run() -> Any:
        return await rec.ainvoke(
            {"messages": [{"role": "user", "content": "go async"}]}, config=cfg
        )

    _ = asyncio.get_event_loop().run_until_complete(_run())
    run_id = rec.last_run_id
    assert run_id is not None

    events = store.list_events(run_id)
    assert len(events) > 0

    # Ensure we captured async_result via a values update and have LLM/TOOL as in sync case
    kinds = {e.action_type for e in events}
    assert ActionType.LLM in kinds and ActionType.TOOL in kinds
    # Values stream should contain async_result from the async node
    saw_async = False
    for e in events:
        if e.labels.get("stream_mode") == "values" and e.output_ref is not None:
            payload = _orjson.loads(store.get_blob(e.output_ref))
            val = payload.get("values") if isinstance(payload, dict) else None
            if isinstance(val, dict) and "async_result" in val:
                saw_async = True
                break
    assert saw_async

    # Staged retrieval tap should persist as RETRIEVAL in async run too
    assert any(e.action_type is ActionType.RETRIEVAL for e in events)
