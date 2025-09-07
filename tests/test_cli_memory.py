from __future__ import annotations

from pathlib import Path
from typing import Any

import orjson as _orjson

from timewarp.cli import main as cli_main
from timewarp.events import Run
from timewarp.store import LocalStore


class _ValuesGraph:
    def __init__(self, updates: list[tuple[str, Any]]) -> None:
        self._updates = updates

    def stream(self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any):
        yield from self._updates


def test_cli_memory_summary_show_and_diff(tmp_path: Path, capsys: Any) -> None:
    # Prepare run with values stream memory under mem.long path
    updates: list[tuple[str, Any]] = [
        ("values", {"values": {"mem": {"long": {"a": 1}}}}),
        ("values", {"values": {"mem": {"long": {"a": 2}}}}),
    ]
    graph = _ValuesGraph(updates)
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="mem-cli", framework="langgraph")

    from timewarp.langgraph import LangGraphRecorder

    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        stream_modes=("values",),
        stream_subgraphs=False,
        memory_paths=("mem.long",),
    )
    _ = rec.invoke({"x": 1}, config={"configurable": {"thread_id": "t-1"}})

    # Summary at step 10 (beyond last)
    rc = cli_main(
        [
            str(store.db_path),
            str(store.blobs_root),
            "memory",
            "summary",
            str(run.run_id),
            "--step",
            "10",
            "--json",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    obj = _orjson.loads(out)
    assert isinstance(obj, dict) and isinstance(obj.get("by_space"), dict)

    # Show snapshot for default space (graph or node)
    rc2 = cli_main(
        [
            str(store.db_path),
            str(store.blobs_root),
            "memory",
            "show",
            str(run.run_id),
            "--step",
            "10",
            "--json",
        ]
    )
    assert rc2 == 0
    out2 = capsys.readouterr().out.strip()
    obj2 = _orjson.loads(out2)
    assert isinstance(obj2, dict) and "by_space" in obj2

    # Diff between two steps
    rc3 = cli_main(
        [
            str(store.db_path),
            str(store.blobs_root),
            "memory",
            "diff",
            str(run.run_id),
            "0",
            "10",
        ]
    )
    assert rc3 == 0
    _ = capsys.readouterr().out.strip()
