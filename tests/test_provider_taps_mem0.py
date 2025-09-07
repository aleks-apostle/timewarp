from __future__ import annotations

from types import ModuleType
from typing import Any
from uuid import UUID

import pytest

from timewarp.bindings import begin_recording_session, bind_memory_taps
from timewarp.events import ActionType, Run
from timewarp.langgraph.taps import flush_provider_taps
from timewarp.store import LocalStore


class _Mem0Memory:
    def __init__(self) -> None:
        self._store: list[Any] = []

    # write ops
    def save(self, item: Any, **_: Any) -> None:
        self._store.append(item)

    def add(self, item: Any, **_: Any) -> None:
        self._store.append(item)

    def delete(self, key: str, **_: Any) -> None:
        return None

    # read ops
    def search(self, query: Any, top_k: int | None = None, **_: Any) -> list[Any]:
        # Return top_k duplicates for shape
        n = top_k if isinstance(top_k, int) else 2
        return [query] * n

    def retrieve(self, query: Any, top_k: int | None = None, **_: Any) -> list[Any]:
        return self.search(query, top_k=top_k)


@pytest.fixture(autouse=True)
def _inject_fake_mem0(monkeypatch: pytest.MonkeyPatch) -> None:
    m = ModuleType("mem0")
    m.Memory = _Mem0Memory
    monkeypatch.setitem(__import__("sys").modules, "mem0", m)


def _drain_taps_to_events(store: LocalStore, run_id: UUID, step: int = 0) -> list[Any]:
    evs: list[Any] = []
    out, step2 = flush_provider_taps(
        store=store,
        run_id=run_id,
        step=step,
        actor="graph",
        namespace_label=None,
        thread_id=None,
        adapter_version="test",
        privacy_marks=None,
        pruner=None,
    )
    for e in out:
        store.append_event(e)
        evs.append(e)
    # drain again to ensure queue is empty
    out2, _ = flush_provider_taps(
        store=store,
        run_id=run_id,
        step=step2,
        actor="graph",
        namespace_label=None,
        thread_id=None,
        adapter_version="test",
        privacy_marks=None,
        pruner=None,
    )
    assert not out2
    return evs


def test_mem0_memory_and_retrieval_taps(tmp_path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="mem0", framework="langgraph")
    store.create_run(run)

    teardown = bind_memory_taps()
    try:
        end_session = begin_recording_session(run.run_id)
        # Use fake mem0 within active session so taps are visible
        from mem0 import Memory  # type: ignore

        client = Memory()
        client.save({"k": 1})
        _ = client.search("hello", top_k=3)
        events = _drain_taps_to_events(store, run.run_id, 0)
        end_session()
    finally:
        teardown()

    # We expect one MEMORY write (PUT) and one RETRIEVAL
    kinds = [e.action_type for e in events]
    assert ActionType.MEMORY in kinds and ActionType.RETRIEVAL in kinds
    # Check retrieval policy/top_k and provider label
    ret = next(e for e in events if e.action_type is ActionType.RETRIEVAL)
    assert ret.mem_provider == "Mem0"
    assert ret.top_k == 3
    # Hashes include results digest
    assert "results" in (ret.hashes or {})
