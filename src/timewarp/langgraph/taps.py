from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

from ..bindings import try_pop_memory_taps
from ..determinism import now as tw_now
from ..events import ActionType, BlobKind, Event, hash_bytes
from ..store import LocalStore
from .anchors import make_anchor_id
from .memory import infer_mem_scope_from_path
from .serialize import normalize_bytes
from .versioning import get_timewarp_version as _get_timewarp_version


def flush_provider_taps(
    *,
    store: LocalStore,
    run_id: UUID,
    step: int,
    actor: str,
    namespace_label: str | None,
    thread_id: str | None,
    adapter_version: str,
    privacy_marks: dict[str, str] | None,
    mem_scope_from_path: Callable[[str], str] | None = None,
    pruner: Callable[[Any], Any] | None = None,
) -> tuple[list[Event], int]:
    events: list[Event] = []
    taps = try_pop_memory_taps()
    if not taps:
        return events, step

    for env in taps:
        kind = env.get("kind")
        try:
            if kind == "MEMORY":
                key = env.get("key")
                value = env.get("value")
                mem_op = env.get("mem_op") or "PUT"
                mem_provider = env.get("mem_provider") or "Custom"
                mem_scope = env.get("mem_scope") or (
                    infer_mem_scope_from_path(str(key)) if isinstance(key, str) else None
                )
                mem_space = env.get("mem_space") or (actor or "graph")
                if pruner is not None:
                    try:
                        value = pruner(value)
                    except Exception:
                        pass
                payload = {"key": key, "value": value}
                data_b = normalize_bytes(payload, privacy_marks=privacy_marks)
                blob = store.put_blob(run_id, step, BlobKind.MEMORY, data_b)
                h = blob.sha256_hex
                labels: dict[str, str] = {}
                if namespace_label:
                    labels["namespace"] = namespace_label
                if thread_id:
                    labels["thread_id"] = thread_id
                if actor and actor != "graph":
                    labels["node"] = actor
                labels["mem_op"] = str(mem_op)
                if mem_scope:
                    labels["mem_scope"] = str(mem_scope)
                labels["mem_space"] = str(mem_space)
                try:
                    labels["anchor_id"] = make_anchor_id(ActionType.MEMORY, actor, labels)
                except Exception:
                    pass
                ev = Event(
                    run_id=run_id,
                    step=step,
                    action_type=ActionType.MEMORY,
                    actor=actor or "graph",
                    output_ref=blob,
                    hashes={"item": h},
                    labels=labels,
                    model_meta={
                        "adapter_version": adapter_version,
                        "timewarp_version": _get_timewarp_version(),
                        "framework": "langgraph",
                        "mem_provider": str(mem_provider),
                    },
                    ts=tw_now(),
                )
                ev = ev.model_copy(update={"mem_provider": str(mem_provider)})
                events.append(ev)
                step += 1
            elif kind == "RETRIEVAL":
                query = env.get("query")
                items = env.get("items")
                policy = env.get("policy") or {}
                if not isinstance(items, list) or not items:
                    continue
                retriever = policy.get("retriever") if isinstance(policy, dict) else None
                top_k = policy.get("top_k") if isinstance(policy, dict) else None
                mem_provider = env.get("mem_provider") or "Custom"
                query_id = env.get("query_id")
                payload = {"query": query, "items": items, "policy": policy}
                data_b = normalize_bytes(payload, privacy_marks=privacy_marks)
                blob = store.put_blob(run_id, step, BlobKind.MEMORY, data_b)
                hashes: dict[str, str] = {}
                try:
                    from ..codec import to_bytes as _to_bytes

                    if query is not None:
                        hashes["query"] = hash_bytes(_to_bytes(query))
                    hashes["results"] = hash_bytes(_to_bytes({"items": items}))
                except Exception:
                    pass
                labels2: dict[str, str] = {}
                if namespace_label:
                    labels2["namespace"] = namespace_label
                if thread_id:
                    labels2["thread_id"] = thread_id
                if actor and actor != "graph":
                    labels2["node"] = actor
                try:
                    labels2["anchor_id"] = make_anchor_id(ActionType.RETRIEVAL, actor, labels2)
                except Exception:
                    pass
                ev2 = Event(
                    run_id=run_id,
                    step=step,
                    action_type=ActionType.RETRIEVAL,
                    actor=actor or "graph",
                    output_ref=blob,
                    hashes=hashes,
                    labels=labels2,
                    model_meta={
                        "adapter_version": adapter_version,
                        "timewarp_version": _get_timewarp_version(),
                        "framework": "langgraph",
                        "mem_provider": str(mem_provider),
                    },
                    ts=tw_now(),
                )
                ev2 = ev2.model_copy(
                    update={
                        "retriever": str(retriever) if isinstance(retriever, str) else None,
                        "top_k": int(top_k) if isinstance(top_k, int) else None,
                        "query_id": str(query_id) if isinstance(query_id, str | int) else None,
                        "mem_provider": str(mem_provider),
                    }
                )
                events.append(ev2)
                step += 1
        except Exception:
            continue
    return events, step
