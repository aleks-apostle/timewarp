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
def test_cli_tools_summary_and_detail(tmp_path: Path, capsys: Any) -> None:
    # Prepare store and record a baseline run (messages+updates to capture LLM)
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)

    graph = make_graph()
    run = Run(project="p", name="cli-tools", framework="langgraph")

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

    # Identify first LLM step (if any)
    events = store.list_events(run.run_id)
    llm_step = next((e.step for e in events if e.action_type is ActionType.LLM), None)
    assert llm_step is not None, "expected at least one LLM event in the run"

    # CLI tools summary (JSON)
    rc = cli_main([str(db), str(blobs), "tools", str(run.run_id), "--json"])
    assert rc == 0
    out = capsys.readouterr().out.strip()
    rows = _orjson.loads(out)
    assert isinstance(rows, list)
    assert any(r.get("step") == llm_step for r in rows)
    # Validate shape for one row
    row0 = rows[0]
    assert set(["step", "actor", "thread_id", "available", "called"]).issubset(row0.keys())

    # CLI tools detail for the LLM step (JSON)
    rc2 = cli_main(
        [str(db), str(blobs), "tools", str(run.run_id), "--step", str(llm_step), "--json"]
    )
    assert rc2 == 0
    out2 = capsys.readouterr().out.strip()
    detail = _orjson.loads(out2)
    assert isinstance(detail, dict)
    assert detail.get("llm_step") == llm_step
    assert "called" in detail and isinstance(detail["called"], list)
    assert "tools_count" in detail
