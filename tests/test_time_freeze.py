from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from timewarp.determinism import freeze_time_at, now
from timewarp.events import ActionType, BlobKind, BlobRef, Event
from timewarp.replay import PlaybackLLM, _EventCursor


def test_freeze_time_context_manager_nested() -> None:
    t1 = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
    t2 = datetime(2024, 1, 1, 0, 0, 1, tzinfo=UTC)
    with freeze_time_at(t1):
        assert now() == t1
        with freeze_time_at(t2):
            assert now() == t2
        # restores to outer
        assert now() == t1


def test_playback_llm_freeze_time_applies_during_output_fetch(tmp_path: Path) -> None:
    # Build a single recorded LLM event with known ts
    run_id = __import__("uuid").uuid4()
    ts = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)
    # Create a real blob ref by writing a small payload
    from timewarp.codec import to_bytes
    from timewarp.store import LocalStore

    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    bref = store.put_blob(run_id, 1, BlobKind.OUTPUT, to_bytes({"ok": True}))
    ev = Event(
        run_id=run_id,
        step=1,
        action_type=ActionType.LLM,
        actor="node",
        output_ref=bref,
        ts=ts,
    )
    cursor = _EventCursor(events=[ev], action_type=ActionType.LLM, start_index=0)

    captured: list[datetime] = []

    class CapturingStore(LocalStore):
        def get_blob(self, ref: BlobRef) -> bytes:  # type: ignore[override]
            captured.append(now())
            return super().get_blob(ref)

    cstore = CapturingStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    llm = PlaybackLLM(store=cstore, cursor=cursor, freeze_time=True)
    _ = llm.invoke("ignored")
    assert captured and captured[0] == ts
