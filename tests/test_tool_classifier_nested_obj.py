from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _Tool:
    def __init__(self) -> None:
        self.name = "echo"
        self.mcp_transport = "stdio"

        class _Server:
            def __init__(self) -> None:
                self.url = "stdio://echo"

        self.server = _Server()


class _GraphWithNestedTool:
    def stream(
        self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any
    ) -> Iterable[Any]:
        update = {"node": "tool_node", "tool": _Tool(), "args": [inputs]}
        yield ("updates", update)


def test_nested_tool_object_classified_as_mcp(tmp_path: Path) -> None:
    graph = _GraphWithNestedTool()
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="nestedtool", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        stream_modes=("updates",),
        stream_subgraphs=False,
    )
    _ = rec.invoke({"x": 1}, config={})
    events = store.list_events(run.run_id)
    tool_evs = [e for e in events if e.action_type is ActionType.TOOL]
    assert tool_evs, "expected at least one TOOL event"
    ev = tool_evs[0]
    assert ev.tool_kind == "MCP"
    assert ev.tool_name == "echo"
    # One of these may be present depending on classifier
    assert (ev.mcp_server is not None) or (ev.mcp_transport is not None)
