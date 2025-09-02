from __future__ import annotations

from pathlib import Path

import pytest

from timewarp.diff import first_divergence
from timewarp.langgraph import wrap
from timewarp.store import LocalStore


@pytest.mark.skipif(
    __import__("importlib").import_module("importlib").util.find_spec("langgraph") is None,  # type: ignore[attr-defined]
    reason="langgraph not installed",
)
def test_parallel_branches_record_and_equivalence(tmp_path: Path) -> None:
    # Import the example factory from file path to avoid sys.path issues
    from importlib import util as _util

    root = Path(__file__).resolve().parents[1]
    mod_path = root / "examples" / "langgraph_demo" / "parallel_app.py"
    spec = _util.spec_from_file_location("tw_parallel_app", str(mod_path))
    assert spec and spec.loader
    mod = _util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    make_graph_parallel = mod.make_graph_parallel

    store = LocalStore(db_path=tmp_path / "tw.sqlite3", blobs_root=tmp_path / "blobs")
    g = make_graph_parallel()
    rec1 = wrap(
        g,
        project="demo",
        name="p1",
        store=store,
        stream_modes=("updates", "values"),
        snapshot_every=10,
        snapshot_on=("terminal", "decision"),
        event_batch_size=10,
    )
    _ = rec1.invoke({"text": "t"}, config={"configurable": {"thread_id": "t-1"}})
    run_a = rec1.last_run_id

    # Record a second run under identical inputs
    rec2 = wrap(
        make_graph_parallel(),
        project="demo",
        name="p2",
        store=store,
        stream_modes=("updates", "values"),
        snapshot_every=10,
        snapshot_on=("terminal", "decision"),
        event_batch_size=10,
    )
    _ = rec2.invoke({"text": "t"}, config={"configurable": {"thread_id": "t-1"}})
    run_b = rec2.last_run_id

    # Expect equivalence by alignment and output hashes
    assert first_divergence(store, run_a, run_b) is None
