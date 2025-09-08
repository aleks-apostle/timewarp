from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import orjson as _orjson
import pytest

from timewarp import messages_pruner, wrap
from timewarp.bindings import stage_memory_tap
from timewarp.cli import main as cli_main
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
def test_e2e_langgraph_native_consolidated(tmp_path: Path, capsys: Any) -> None:
    """Consolidated end-to-end test exercising LangGraph + Timewarp features.

    Covers: multi-agent orchestration, MCP-like tools, parallel exec, HITL,
    snapshots, memory/retrieval taps, CLI tools+memory views, programmatic
    resume, overlay injection, fork+record, and diff analysis.
    """

    # Isolated store
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)

    # Build the multi-agent demo graph (sync variant for deterministic invoke)
    graph = make_graph_multi(include_async=False)

    # Stage provider taps to persist MEMORY and RETRIEVAL events end-to-end
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
            "query": "consolidated",
            "items": [
                {"id": "a", "text": "A", "score": 0.9},
                {"id": "b", "text": "B", "score": 0.7},
            ],
            "policy": {"retriever": "vector", "top_k": 2},
            "query_id": "qid-consolidated",
        }
    )

    # Record baseline run via facade to enable taps and redaction
    rec = wrap(
        graph,
        project="demo",
        name="consolidated-e2e",
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
    cfg: dict[str, Any] = {"configurable": {"thread_id": "t-e2e"}}
    result = rec.invoke({"messages": [{"role": "user", "content": "run e2e"}]}, config=cfg)
    run_id = cast(Any, rec.last_run_id)

    # Sanity: result contains a report from compose node
    assert isinstance(result, dict)
    res_vals = cast(dict[str, Any], result.get("values", result))
    assert "report" in res_vals

    # Verify events and kinds
    events = store.list_events(run_id)
    assert len(events) > 0
    steps = [e.step for e in events]
    assert steps == sorted(steps)  # respect serialized completion order

    kinds = {e.action_type for e in events}
    for k in (
        ActionType.SYS,
        ActionType.LLM,
        ActionType.TOOL,
        ActionType.DECISION,
        ActionType.HITL,
        ActionType.SNAPSHOT,
        ActionType.MEMORY,
        ActionType.RETRIEVAL,
    ):
        assert k in kinds

    # LLM event has prompt hashing or prompt_ctx hashing; includes framework meta
    llm_ev = next(e for e in events if e.action_type is ActionType.LLM)
    assert llm_ev.hashes.get("prompt") is not None or llm_ev.hashes.get("prompt_ctx") is not None
    assert isinstance(llm_ev.model_meta, dict) and llm_ev.model_meta.get("framework") == "langgraph"

    # TOOL classification, args hash, privacy redaction persisted
    tool_ev = next(e for e in events if e.action_type is ActionType.TOOL)
    assert tool_ev.tool_kind == "MCP" and tool_ev.tool_name == "calc"
    assert isinstance(tool_ev.hashes.get("args"), str)
    assert tool_ev.output_ref is not None
    payload = _orjson.loads(store.get_blob(tool_ev.output_ref))
    if isinstance(payload, dict):
        kp = (
            payload.get("kwargs")
            if "kwargs" in payload
            else (
                payload.get("tool_args", {}).get("kwargs")
                if isinstance(payload.get("tool_args"), dict)
                else None
            )
        )
        if isinstance(kp, dict) and "ssn" in kp:
            assert (
                isinstance(kp["ssn"], str)
                and kp["ssn"].endswith("6789")
                and kp["ssn"].startswith("***")
            )

    # RETRIEVAL tap present & decoded content
    ret_ev = next(e for e in events if e.action_type is ActionType.RETRIEVAL)
    rp = _orjson.loads(store.get_blob(ret_ev.output_ref))
    assert (
        ret_ev.retriever == "vector" and ret_ev.top_k == 2 and ret_ev.query_id == "qid-consolidated"
    )
    assert isinstance(rp.get("items"), list) and len(cast(list[Any], rp.get("items"))) == 2

    # CLI tools summary and detail
    rc = cli_main([str(db), str(blobs), "tools", str(run_id), "--json"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    rows = _orjson.loads(out)
    assert isinstance(rows, list) and any(r.get("step") == llm_ev.step for r in rows)
    rc2 = cli_main(
        [str(db), str(blobs), "tools", str(run_id), "--step", str(llm_ev.step), "--json"]
    )
    assert rc2 == 0
    detail = _orjson.loads(capsys.readouterr().out.strip())
    assert isinstance(detail, dict) and detail.get("llm_step") == llm_ev.step

    # CLI memory views (summary/show) should work using MEMORY taps
    rc3 = cli_main(
        [str(db), str(blobs), "memory", "summary", str(run_id), "--step", str(max(steps)), "--json"]
    )
    assert rc3 == 0
    mem_summary = _orjson.loads(capsys.readouterr().out.strip())
    assert isinstance(mem_summary, dict) and isinstance(mem_summary.get("by_space"), dict)
    rc4 = cli_main(
        [str(db), str(blobs), "memory", "show", str(run_id), "--step", str(max(steps)), "--json"]
    )
    assert rc4 == 0
    mem_show = _orjson.loads(capsys.readouterr().out.strip())
    assert isinstance(mem_show, dict) and "by_space" in mem_show

    # Exporter serializes run + events with optional inline blobs
    export_payload = serialize_run(store, run_id, include_blobs=True)
    assert "run" in export_payload and "events" in export_payload
    assert isinstance(export_payload["events"], list)

    # Deterministic resume using recorded outputs via Replay.resume
    factory = "examples.langgraph_demo.multi_agent_full:make_graph_multi"
    sess = Replay.resume(
        store, app_factory=factory, run_id=run_id, thread_id="t-e2e", freeze_time=True
    )
    assert sess.result is None or isinstance(sess.result, dict)

    # Replay overlay injection: tweak a 'values' report
    tgt = None
    for e in events:
        if e.labels.get("stream_mode") == "values" and e.output_ref is not None:
            obj = _orjson.loads(store.get_blob(e.output_ref))
            vv = obj.get("values") if isinstance(obj, dict) else None
            if isinstance(vv, dict) and "report" in vv:
                tgt = (e.step, obj)
                break
    assert tgt is not None
    step_to_inject, orig_payload = tgt
    new_payload = _orjson.loads(_orjson.dumps(orig_payload))
    if isinstance(new_payload, dict) and isinstance(new_payload.get("values"), dict):
        new_payload["values"]["report"] = "[override]"
    rp = Replay(store=store, run_id=run_id)
    rp.goto(step_to_inject + 1)
    before = cast(dict[str, Any] | None, rp.inspect_state())
    assert isinstance(before, dict)
    prev_report = cast(str, before.get("report"))
    rp.inject(step_to_inject, new_payload)
    rp.goto(step_to_inject + 1)
    after = cast(dict[str, Any] | None, rp.inspect_state())
    assert isinstance(after, dict)
    assert after.get("report") == "[override]" or after.get("report") != prev_report

    # Fork what-if by overriding first LLM step and record the branch
    from timewarp.events import Run as _Run
    from timewarp.langgraph import LangGraphRecorder
    from timewarp.replay import LangGraphReplayer

    llm_step = llm_ev.step
    graph2 = make_graph_multi(include_async=False)
    replayer = LangGraphReplayer(graph=graph2, store=store)

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
        thread_id="t-e2e",
        install_wrappers=_installer,
        freeze_time=True,
    )

    try:
        # Recover original input and record the fork under provided id
        orig_input: dict[str, Any] | None = None
        for e in events:
            if e.input_ref is not None:
                orig_input = cast(dict[str, Any], _orjson.loads(store.get_blob(e.input_ref)))
                break
        assert isinstance(orig_input, dict)
        fork_run = _Run(
            run_id=new_run_id,
            project="demo",
            name="consolidated-e2e-fork",
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

        # Diff baseline vs fork
        fd = first_divergence(store, run_id, new_run_id, window=5)
        assert fd is not None and fd.reason == "output hash mismatch" and fd.step_a == llm_step
        bi = bisect_divergence(store, run_id, new_run_id, window=5)
        assert bi is not None and cast(dict[str, Any], bi).get("cause") in {
            "output hash mismatch",
            "anchor mismatch",
        }

        # CLI diff JSON should reflect mismatch at LLM step as well
        rc5 = cli_main([str(db), str(blobs), "diff", str(run_id), str(new_run_id), "--json"])
        assert rc5 == 0
        diff_obj = _orjson.loads(capsys.readouterr().out.strip())
        assert diff_obj.get("reason") in {"output hash mismatch", "anchor mismatch"}
        assert diff_obj.get("step_a") == llm_step
    finally:
        for td in teardowns:
            try:
                td()
            except Exception:
                pass
