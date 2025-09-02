"""Optional LangSmith exporter stub.

This module provides a minimal serialization helper that prepares a run and its
events for export to external systems (e.g., LangSmith). It avoids hard
dependencies; callers can pass in their own client and perform the upload.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from ..codec import from_bytes
from ..events import Event
from ..store import LocalStore


def serialize_run(
    store: LocalStore, run_id: UUID, *, include_blobs: bool = False
) -> dict[str, Any]:
    """Serialize run metadata and events to a JSON-like dict.

    - When include_blobs is True, small blobs are inlined as JSON payloads when
      possible; otherwise only BlobRef metadata is included.
    """
    runs = {r.run_id: r for r in store.list_runs()}
    run = runs.get(run_id)
    if run is None:
        raise ValueError(f"unknown run_id: {run_id}")
    events: list[Event] = store.list_events(run_id)
    evs_payload: list[dict[str, Any]] = []
    for e in events:
        item = e.model_dump(mode="json")
        if include_blobs:
            try:
                if e.input_ref is not None:
                    item["input_payload"] = from_bytes(store.get_blob(e.input_ref))
                if e.output_ref is not None:
                    item["output_payload"] = from_bytes(store.get_blob(e.output_ref))
            except Exception:
                # Best-effort: skip blobs that fail to load
                pass
        evs_payload.append(item)
    return {
        "run": run.model_dump(mode="json"),
        "events": evs_payload,
    }


def export_run(store: LocalStore, run_id: UUID, *, client: Any | None = None) -> dict[str, Any]:
    """Prepare export payload; if a client is provided, invoke a best-effort upload.

    The exact client API is left to the caller to avoid hard dependencies. The
    returned payload can be written to disk or used directly by custom tooling.
    """
    payload = serialize_run(store, run_id, include_blobs=False)
    # If a client is provided, perform a best-effort send using common conventions
    try:
        if client is not None:
            # Expect a generic 'create_run' method; ignore errors for portability
            send = getattr(client, "create_run", None)
            if callable(send):
                send(payload)
    except Exception:
        # Silent best-effort; callers can handle errors explicitly if needed
        pass
    return payload
