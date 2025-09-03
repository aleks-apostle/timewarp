"""LangSmith-friendly export helpers.

Purpose
- Provide JSON-like payloads suitable for sending to LangSmith or similar tools.
- Avoid hard dependencies; callers supply their own client object.

Mapping
- ``serialize_run`` returns a dict with two keys:
  - ``run``: ``Run.model_dump(mode="json")`` — run metadata only.
  - ``events``: list of ``Event.model_dump(mode="json")`` for each event.
    When ``include_blobs=True``, the following keys may be inlined per event:
    - ``input_payload``: decoded JSON for ``input_ref`` blob if present/JSON.
    - ``output_payload``: decoded JSON for ``output_ref`` blob if present/JSON.

Caveats
- Redaction/encryption occur before persistence; exported payload reflects stored bytes.
- Hashes are computed over stored (possibly redacted/encrypted) content.
- Token chunks from messages aggregation are referenced via ``chunks_ref`` only.
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

    - ``include_blobs=True`` attempts to inline small JSON blobs as
      ``input_payload``/``output_payload`` fields; otherwise only BlobRef
      metadata is included on events.
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
    """Prepare export payload; optionally invoke a client to upload.

    - If ``client`` is provided and exposes ``create_run(payload)``, it is called
      best-effort with the serialized payload. Errors are swallowed intentionally
      to avoid hard coupling to external SDKs.
    - The returned payload can be written to disk or consumed by custom tools.
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
