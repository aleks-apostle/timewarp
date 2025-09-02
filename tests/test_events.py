from __future__ import annotations

from datetime import UTC
from uuid import uuid4

from timewarp.events import ActionType, BlobKind, BlobRef, Event, Run


def test_event_canonical_hash_stable() -> None:
    run_id = uuid4()
    blob = BlobRef(
        run_id=run_id,
        step=0,
        kind=BlobKind.INPUT,
        path="runs/x/events/0/input.bin",
        size_bytes=10,
        compression="zstd",
        sha256_hex="00" * 32,
        content_type="application/json",
    )
    e1 = Event(
        run_id=run_id,
        step=0,
        action_type=ActionType.SYS,
        actor="graph",
        input_ref=blob,
    )
    e2 = Event(
        run_id=run_id,
        step=0,
        action_type=ActionType.SYS,
        actor="graph",
        input_ref=blob,
        ts=e1.ts,  # pin to same timestamp
    )
    assert e1.canonical_bytes() == e2.canonical_bytes()
    assert e1.sha256_hex() == e2.sha256_hex()


def test_run_defaults() -> None:
    r = Run(project="proj", name="name")
    assert r.started_at.tzinfo is UTC
