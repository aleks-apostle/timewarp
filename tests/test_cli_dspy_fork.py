from __future__ import annotations

from pathlib import Path

import orjson as _orjson
import pytest

from timewarp.cli import main as cli_main
from timewarp.events import Run
from timewarp.store import LocalStore

try:
    from examples.langgraph_demo.app import make_graph

    LANGGRAPH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dep environment
    LANGGRAPH_AVAILABLE = False


@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="langgraph not installed")
def test_cli_dspy_fork_prompts_record(tmp_path: Path) -> None:
    # Record a baseline run
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)

    graph = make_graph()
    run = Run(project="p", name="baseline", framework="langgraph")
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

    # Overrides: add a system message for the 'compose' node
    overrides_path = tmp_path / "overrides.json"
    overrides_path.write_bytes(
        _orjson.dumps({"compose": {"mode": "prepend_system", "text": "DSPy override"}})
    )

    rc = cli_main(
        [
            str(db),
            str(blobs),
            "dspy",
            "fork",
            str(run.run_id),
            "--app",
            "examples.langgraph_demo.app:make_graph",
            "--overrides",
            str(overrides_path),
            "--thread",
            "t-1",
            "--allow-diff",
            "--record-fork",
        ]
    )
    assert rc == 0

    # Find the forked run
    runs = {r.run_id: r for r in store.list_runs()}
    fork_id = None
    for r_id, r in runs.items():
        if r.labels.get("branch_of") == str(run.run_id):
            fork_id = r_id
            break
    assert fork_id is not None
