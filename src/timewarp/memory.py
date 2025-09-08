from __future__ import annotations

from typing import Any
from uuid import UUID

from .codec import from_bytes
from .events import ActionType
from .store import LocalStore


def rebuild_memory_snapshot(
    store: LocalStore, run_id: UUID, step: int, thread_id: str | None = None
) -> dict[str, Any]:
    """Reconstruct a per-agent memory snapshot up to (and including) step.

    Aggregates MEMORY writes into scoped maps (short/working/long) by mem_space, and
    collects RETRIEVAL events as a history list per mem_space.

    The returned object is JSON-serializable.
    """
    events = store.list_events(run_id)
    # Initialize per-space aggregates
    by_space: dict[str, dict[str, Any]] = {}

    def ensure_space(space: str) -> dict[str, Any]:
        if space not in by_space:
            by_space[space] = {
                "short": {},
                "working": {},
                "long": {},
                "retrievals": [],
            }
        return by_space[space]

    for e in events:
        if e.step > step:
            break
        if thread_id is not None and e.labels.get("thread_id") != thread_id:
            continue
        if e.action_type is ActionType.MEMORY:
            space = e.labels.get("mem_space") or e.mem_space or (e.labels.get("node") or e.actor)
            scope = e.labels.get("mem_scope") or e.mem_scope or "working"
            if not space:
                space = "graph"
            dest = ensure_space(str(space))
            try:
                if e.output_ref is None:
                    continue
                payload = from_bytes(store.get_blob(e.output_ref))
                if not isinstance(payload, dict):
                    continue
                key = payload.get("key")
                if not isinstance(key, str):
                    continue
                mem_op = e.labels.get("mem_op") or e.mem_op or "PUT"
                if mem_op.upper() in ("DELETE", "EVICT"):
                    try:
                        # Remove key when present
                        if key in dest.get(scope, {}):
                            del dest[scope][key]
                    except Exception:
                        pass
                else:
                    dest[scope][key] = payload.get("value")
            except Exception:
                continue
        elif e.action_type is ActionType.RETRIEVAL:
            space = e.labels.get("mem_space") or e.mem_space or (e.labels.get("node") or e.actor)
            if not space:
                space = "graph"
            dest = ensure_space(str(space))
            try:
                if e.output_ref is None:
                    continue
                payload = from_bytes(store.get_blob(e.output_ref))
                if not isinstance(payload, dict):
                    continue
                entry = {
                    "step": e.step,
                    "mem_provider": e.mem_provider or (e.model_meta or {}).get("mem_provider"),
                    "query": payload.get("query"),
                    "policy": payload.get("policy"),
                    "items_count": len(payload.get("items", []))
                    if isinstance(payload.get("items"), list)
                    else 0,
                    "anchor_id": e.labels.get("anchor_id"),
                }
                dest["retrievals"].append(entry)
            except Exception:
                continue

    return {"by_space": by_space, "up_to_step": int(step)}
