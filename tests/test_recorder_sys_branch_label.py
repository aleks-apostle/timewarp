from __future__ import annotations

from pathlib import Path

from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _FakeUpdatesGraph:
    def __init__(self) -> None:
        self._updates: list[tuple[str, object]] = []

    def stream(
        self, inputs: dict[str, object], config: dict[str, object] | None = None, **_: object
    ):
        # Produce no updates; we're testing the initial SYS emission only
        if False:  # pragma: no cover - keep generator shape
            yield from self._updates
        return


def test_recorder_initial_sys_carries_branch_of_label(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    parent = Run(project="p", name="orig")
    store.create_run(parent)
    branch_label = str(parent.run_id)
    run = Run(project="p", name="fork", framework="langgraph", labels={"branch_of": branch_label})

    rec = LangGraphRecorder(
        graph=_FakeUpdatesGraph(),
        store=store,
        run=run,
        snapshot_every=0,
        stream_modes=("updates",),
        stream_subgraphs=False,
    )

    _ = rec.invoke({"x": 0}, config={})

    events = store.list_events(run.run_id)
    assert events
    first = events[0]
    assert first.action_type is ActionType.SYS
    assert first.labels.get("branch_of") == branch_label
