from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson as _orjson
import pytest

from timewarp.cli import main as cli_main
from timewarp.events import ActionType, Run
from timewarp.store import LocalStore

try:
    from examples.langgraph_demo.app import make_graph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep environment
    LANGGRAPH_AVAILABLE = False


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="langgraph not installed")
def test_cli_resume_and_inject_record_fork_end_to_end(tmp_path: Path, capsys) -> None:
    # Prepare store and record a baseline run using recorder (messages+updates to capture LLM)
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)

    graph = make_graph()
    run = Run(project="p", name="e2e", framework="langgraph")

    from timewarp.langgraph import LangGraphRecorder

    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        snapshot_every=0,
        stream_modes=("messages", "updates"),
        stream_subgraphs=True,
    )
    _ = rec.invoke({"text": "hi"}, config={"configurable": {"thread_id": "t-1"}})

    # Find first LLM step to target for injection
    events = store.list_events(run.run_id)
    llm_step = next(e.step for e in events if e.action_type is ActionType.LLM)

    # CLI resume from start with factory app
    rc = cli_main(
        [
            str(db),
            str(blobs),
            "resume",
            str(run.run_id),
            "--from",
            "0",
            "--thread",
            "t-1",
            "--app",
            "examples.langgraph_demo.app:make_graph",
        ]
    )
    assert rc == 0

    # Write replacement payload for the LLM step
    repl = tmp_path / "replacement.json"
    repl.write_bytes(_orjson.dumps({"message": {"content": "OVERRIDE"}}))

    # CLI inject with --record-fork to execute and persist branch
    rc2 = cli_main(
        [
            str(db),
            str(blobs),
            "inject",
            str(run.run_id),
            str(llm_step),
            "--output",
            str(repl),
            "--thread",
            "t-1",
            "--app",
            "examples.langgraph_demo.app:make_graph",
            "--record-fork",
        ]
    )
    assert rc2 == 0

    # Locate the new forked run via branch_of label
    runs = {r.run_id: r for r in store.list_runs()}
    fork_id = None
    for r_id, r in runs.items():
        if r.labels.get("branch_of") == str(run.run_id):
            fork_id = r_id
            break
    assert fork_id is not None

    # CLI diff JSON should show output hash mismatch at the LLM step
    rc3 = cli_main([str(db), str(blobs), "diff", str(run.run_id), str(fork_id), "--json"])
    assert rc3 == 0
    out = capsys.readouterr().out.strip()
    diff_obj: dict[str, Any] = _orjson.loads(out)
    assert diff_obj.get("reason") == "output hash mismatch"
    assert diff_obj.get("step_a") == llm_step
