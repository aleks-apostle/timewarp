from __future__ import annotations

from pathlib import Path

import orjson as _orjson

from timewarp.cli import main as cli_main
from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.store import LocalStore


def test_cli_events_tool_filters(tmp_path: Path, capsys) -> None:
    db = tmp_path / "db.sqlite"
    blobs = tmp_path / "blobs"
    store = LocalStore(db_path=db, blobs_root=blobs)
    run = Run(project="p", name="toolfilters")
    store.create_run(run)

    # SYS
    ib = store.put_blob(run.run_id, 0, BlobKind.INPUT, _orjson.dumps({"in": 1}))
    store.append_event(
        Event(
            run_id=run.run_id,
            step=0,
            action_type=ActionType.SYS,
            actor="graph",
            input_ref=ib,
            hashes={"input": ib.sha256_hex},
        )
    )
    # TOOL with MCP metadata
    tb = store.put_blob(run.run_id, 1, BlobKind.OUTPUT, _orjson.dumps({"ok": True}))
    store.append_event(
        Event(
            run_id=run.run_id,
            step=1,
            action_type=ActionType.TOOL,
            actor="use_mcp",
            output_ref=tb,
            hashes={"output": tb.sha256_hex},
            tool_kind="MCP",
            tool_name="echo",
            mcp_server="stdio://echo",
            mcp_transport="stdio",
        )
    )

    rc = cli_main(
        [
            str(db),
            str(blobs),
            "events",
            str(run.run_id),
            "--type",
            "TOOL",
            "--tool-kind",
            "MCP",
            "--tool-name",
            "echo",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    rows = _orjson.loads(out)
    assert isinstance(rows, list)
    assert len(rows) == 1
    assert rows[0]["type"] == "TOOL"
