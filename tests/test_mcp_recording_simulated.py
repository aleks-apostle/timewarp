from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _FakeGraph:
    """Minimal graph-like object emitting an MCP-like tool update in 'updates' mode.

    This simulates a LangGraph stream update for a tool call without requiring
    any external dependencies. The recorder should classify it as a TOOL event
    and capture MCP metadata via heuristics.
    """

    def stream(
        self, inputs: dict[str, Any], config: dict[str, Any], **kwargs: Any
    ) -> Iterable[tuple[str, Any]]:
        # Emit a single 'updates' tuple with tool metadata and a node name
        update = {
            "node": "use_mcp",
            "tool_kind": "MCP",
            "tool_name": "echo",
            "mcp_server": "stdio://echo",
            "mcp_transport": "stdio",
            "args": [inputs],
        }
        yield ("updates", update)


def test_recorder_classifies_mcp_tool_from_updates(tmp_path: Path) -> None:
    graph = _FakeGraph()
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="mcp-sim", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        snapshot_every=0,
        stream_modes=("updates",),
        stream_subgraphs=False,
    )
    _ = rec.invoke({"text": "hi"}, config={"configurable": {"thread_id": "t-1"}})

    events = store.list_events(run.run_id)
    tools = [e for e in events if e.action_type is ActionType.TOOL]
    assert tools, "expected at least one TOOL event"
    ev = tools[0]
    assert ev.tool_kind == "MCP"
    assert isinstance(ev.tool_name, str) and ev.tool_name
    assert (ev.mcp_server is not None) or (ev.mcp_transport is not None)
