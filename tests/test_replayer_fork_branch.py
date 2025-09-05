from __future__ import annotations

from pathlib import Path

from timewarp.events import Run
from timewarp.replay import LangGraphReplayer
from timewarp.store import LocalStore


class _DummyGraph:
    # No-op graph placeholder for installers binding
    pass


def test_fork_creates_branch_run_with_label(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    orig = Run(project="p", name="orig", framework="langgraph")
    store.create_run(orig)

    graph = _DummyGraph()
    replayer = LangGraphReplayer(graph=graph, store=store)

    def installer(_llm: object, _tool: object, _memory: object) -> None:
        # no-op binding in test
        return None

    new_id = replayer.fork_with_injection(
        orig.run_id, at_step=1, replacement={"x": 1}, thread_id=None, install_wrappers=installer
    )
    assert new_id is not None
    # Verify the new run exists and carries branch_of label
    runs = {r.run_id: r for r in store.list_runs()}
    assert new_id in runs
    assert runs[new_id].labels.get("branch_of") == str(orig.run_id)
