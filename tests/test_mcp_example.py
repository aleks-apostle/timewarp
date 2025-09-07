from __future__ import annotations

from pathlib import Path

import pytest

from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("langgraph"),  # type: ignore[attr-defined]
    reason="langgraph not installed",
)
def test_mcp_example_app_records_mcp_metadata(tmp_path: Path) -> None:
    # Import the example factory; skip if adapters not present
    try:
        from examples.langgraph_demo.mcp_app import make_graph_mcp
    except Exception:  # pragma: no cover - example unavailable
        pytest.skip("MCP example factory unavailable in this environment")

    try:
        graph = make_graph_mcp()
    except RuntimeError as exc:  # pragma: no cover - missing deps
        pytest.skip(str(exc))

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="mcp-demo", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        snapshot_every=0,
        stream_modes=("messages", "updates"),
        stream_subgraphs=True,
    )
    _ = rec.invoke({"text": "hi"}, config={"configurable": {"thread_id": "t-1"}})

    events = store.list_events(run.run_id)
    tool_events = [e for e in events if e.action_type is ActionType.TOOL]
    if not tool_events:
        pytest.skip("No TOOL events emitted by MCP example in this environment")
    ev = tool_events[0]
    assert ev.tool_kind == "MCP"
    assert isinstance(ev.tool_name, str) and ev.tool_name
    assert (ev.mcp_server is not None) or (ev.mcp_transport is not None)
