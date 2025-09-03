from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from timewarp.diff import first_divergence
from timewarp.langgraph import wrap
from timewarp.store import LocalStore


def _has_astream(graph: Any) -> bool:
    return hasattr(graph, "astream") and callable(graph.astream)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("langgraph") is None,  # type: ignore[attr-defined]
    reason="langgraph not installed",
)
def test_async_parity_updates_values_labels_hashes(tmp_path: Path) -> None:
    try:
        from examples.langgraph_demo.app import make_graph  # type: ignore
    except ModuleNotFoundError:
        # Fallback to path-based import when repo root isn't on sys.path
        import importlib.util
        from pathlib import Path as _P

        root = _P(__file__).resolve().parents[1]
        mod_path = root / "examples" / "langgraph_demo" / "app.py"
        spec = importlib.util.spec_from_file_location("_lg_app", mod_path)
        assert spec and spec.loader
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        make_graph = m.make_graph

    graph = make_graph()
    if not _has_astream(graph):
        pytest.skip("graph does not expose .astream")

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    handle = wrap(
        graph,
        project="async-parity",
        store=store,
        stream_modes=("updates", "values"),
        snapshot_every=5,
        stream_subgraphs=True,
        require_thread_id=False,
    )

    cfg = {"configurable": {"thread_id": "t-1"}}

    # Sync run
    _ = handle.invoke({"text": "hi"}, config=cfg)
    run_sync = handle.last_run_id
    assert run_sync is not None

    # Async run
    import asyncio

    async def _run_async() -> Any:
        return await handle.ainvoke({"text": "hi"}, config=cfg)

    _ = asyncio.get_event_loop().run_until_complete(_run_async())
    run_async = handle.last_run_id
    assert run_async is not None

    div = first_divergence(store, run_sync, run_async)
    assert div is None


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("langgraph") is None,  # type: ignore[attr-defined]
    reason="langgraph not installed",
)
def test_async_messages_mode_aggregation(tmp_path: Path) -> None:
    # Ensure fake chat model is available; else skip like other tests
    try:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel  # noqa: F401
    except Exception:
        pytest.skip("langchain-core not available")

    try:
        from examples.langgraph_demo.app import make_graph  # type: ignore
    except ModuleNotFoundError:
        import importlib.util
        from pathlib import Path as _P

        root = _P(__file__).resolve().parents[1]
        mod_path = root / "examples" / "langgraph_demo" / "app.py"
        spec = importlib.util.spec_from_file_location("_lg_app", mod_path)
        assert spec and spec.loader
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        make_graph = m.make_graph

    graph = make_graph()
    if not _has_astream(graph):
        pytest.skip("graph does not expose .astream")

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    handle = wrap(
        graph,
        project="async-msgs",
        store=store,
        stream_modes=("messages",),
        snapshot_every=0,
        stream_subgraphs=False,
    )

    import asyncio

    _ = asyncio.get_event_loop().run_until_complete(
        handle.ainvoke({"text": "ok"}, config={"configurable": {"thread_id": "t-msg"}})
    )
    run_id = handle.last_run_id
    assert run_id is not None
    events = store.list_events(run_id)
    # Expect at least one LLM event aggregated with messages mode
    assert any(
        e.action_type.value == "LLM"
        and e.labels.get("stream_mode") == "messages"
        and isinstance(e.output_ref, object)
        for e in events[1:]
    )


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("langgraph") is None,  # type: ignore[attr-defined]
    reason="langgraph not installed",
)
def test_async_skip_when_astream_missing(tmp_path: Path) -> None:
    # Build a minimal fake graph with .stream only
    class _FakeGraph:
        def stream(self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any):
            yield ("updates", {"ok": True})

    graph = _FakeGraph()
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    handle = wrap(
        graph,
        project="skip-async",
        store=store,
        stream_modes=("updates",),
        snapshot_every=0,
        stream_subgraphs=False,
    )

    # ainvoke should raise since .astream is not available; test ensures we can skip
    with pytest.raises(RuntimeError):
        import asyncio

        asyncio.get_event_loop().run_until_complete(handle.ainvoke({"x": 1}, config={}))
