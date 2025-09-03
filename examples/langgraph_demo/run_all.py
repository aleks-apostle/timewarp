"""End-to-end demo that exercises Timewarp features with the multi-agent graph.

This script will:
- Build a realistic multi-agent LangGraph with mock LLM + tools
- Record a run with full streaming and snapshots
- Resume deterministically from the recorded checkpoints
- Fork the run by injecting a what-if override and record the branch
- Compute and print both first divergence and minimal failing window between runs

Usage:
  uv run python -m examples.langgraph_demo.run_all
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import UUID

from timewarp import messages_pruner, wrap
from timewarp.adapters.installers import bind_langgraph_playback
from timewarp.diff import bisect_divergence, first_divergence
from timewarp.replay import LangGraphReplayer, Replay
from timewarp.store import LocalStore


def main() -> int:
    # Import the richer multi-agent example
    from examples.langgraph_demo.multi_agent_full import make_graph_multi

    # 1) Build + record a synchronous run
    graph = make_graph_multi(include_async=False)
    # Use a dedicated demo store to avoid clashing with prior local schemas
    store = LocalStore(db_path=Path("tw_runs/demo.sqlite3"), blobs_root=Path("tw_runs/blobs"))
    rec = wrap(
        graph,
        project="demo",
        name="multi-agent-debug",
        store=store,
        snapshot_every=5,
        snapshot_on=("terminal", "decision"),
        stream_modes=("updates", "messages", "values"),
        state_pruner=messages_pruner(max_len=500, max_items=50),
        enable_record_taps=True,
        event_batch_size=20,
        privacy_marks={"kwargs.ssn": "mask4"},
    )
    cfg: dict[str, Any] = {"configurable": {"thread_id": "t-demo"}}
    _ = rec.invoke({"messages": [{"role": "user", "content": "do work"}]}, config=cfg)
    base_id = rec.last_run_id
    print(f"Recorded base run_id: {base_id}")

    # 2) Programmatic resume from nearest checkpoint (deterministic; no side-effects)
    # Programmatic resume from nearest checkpoint (deterministic; no side-effects)
    _ = Replay.resume(
        store,
        app_factory="examples.langgraph_demo.multi_agent_full:make_graph_multi",
        run_id=UUID(str(base_id)),
        from_step=None,
        thread_id="t-demo",
        strict_meta=False,
        freeze_time=True,
    )

    # 3) Fork with injection by overriding the first TOOL/LLM output and record the branch
    # Prepare and record a what-if branch by overriding the first TOOL or LLM event
    events = store.list_events(UUID(str(base_id)))
    target_step: int | None = next((e.step for e in events if e.action_type.value == "TOOL"), None)
    if target_step is None:
        target_step = next((e.step for e in events if e.action_type.value == "LLM"), None)
    if target_step is None:
        print("No TOOL/LLM event found for injection; exiting")
        return 0
    # Prepare replayer and override
    replayer = LangGraphReplayer(graph=graph, store=store)
    teardowns: list[Any] = []

    def _installer(llm: Any, tool: Any) -> None:
        td = bind_langgraph_playback(graph, llm, tool)
        teardowns.append(td)

    alt_output = {"message": {"content": "[what-if override]"}}
    new_id = replayer.fork_with_injection(
        UUID(str(base_id)),
        target_step,
        alt_output,
        thread_id="t-demo",
        install_wrappers=_installer,
        freeze_time=True,
    )
    # Record the forked execution under new_id
    # Recover original input
    orig_input: Any | None = None
    for e in events:
        if e.input_ref is not None:
            from timewarp.codec import from_bytes as _from_bytes

            orig_input = _from_bytes(store.get_blob(e.input_ref))
            break
    if orig_input is not None:
        from timewarp.adapters.langgraph import LangGraphRecorder
        from timewarp.events import Run

        new_run = Run(
            run_id=new_id,
            project="demo",
            name="multi-agent-debug-fork",
            framework="langgraph",
            labels={"branch_of": str(base_id), "override_step": str(target_step)},
        )
        rec2 = LangGraphRecorder(
            graph=graph,
            store=store,
            run=new_run,
            stream_modes=("updates", "messages", "values"),
            stream_subgraphs=True,
            snapshot_every=5,
            snapshot_on={"terminal", "decision"},
        )
        _ = rec2.invoke(orig_input, config=cfg)
    # Teardown installers
    for td in teardowns:
        try:
            td()
        except Exception:
            pass
    fork_id = new_id
    print(f"Recorded forked run_id: {fork_id}")

    # 4) Compute diffs between base and fork
    d1 = first_divergence(store, UUID(str(base_id)), UUID(str(fork_id)), window=5)
    win = bisect_divergence(store, UUID(str(base_id)), UUID(str(fork_id)), window=5)

    try:
        import orjson as _orjson

        payload: dict[str, Any] = {
            "base_run": str(base_id),
            "fork_run": str(fork_id),
            "first_divergence": None
            if d1 is None
            else {"step_a": d1.step_a, "step_b": d1.step_b, "reason": d1.reason},
            "minimal_window": win,
        }
        print(_orjson.dumps(payload).decode("utf-8"))
    except Exception:
        import json as _json

        payload2: dict[str, Any] = {
            "base_run": str(base_id),
            "fork_run": str(fork_id),
            "first_divergence": None
            if d1 is None
            else {"step_a": d1.step_a, "step_b": d1.step_b, "reason": d1.reason},
            "minimal_window": win,
        }
        print(_json.dumps(payload2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
