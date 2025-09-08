from __future__ import annotations

from typing import Any
from uuid import UUID

from ..determinism import now as tw_now
from ..events import ActionType, BlobKind, Event, hash_bytes
from ..store import LocalStore
from .anchors import make_anchor_id
from .serialize import normalize_bytes
from .versioning import get_timewarp_version as _get_timewarp_version


def detect_retrieval(values_like: Any) -> dict[str, Any] | None:
    try:
        src = values_like
        if isinstance(src, dict) and "values" in src and isinstance(src["values"], dict):
            src = src["values"]
        if not isinstance(src, dict):
            return None
        env: dict[str, Any] | None = None
        cand = src.get("retrieval")
        if isinstance(cand, dict):
            env = dict(cand)
        if env is None:
            if "query" in src and isinstance(src.get("results"), list):
                env = {"query": src.get("query"), "items": src.get("results")}
            elif "query" in src and isinstance(src.get("documents"), list):
                env = {"query": src.get("query"), "items": src.get("documents")}
            elif isinstance(src.get("docs"), list) and src.get("query") is not None:
                env = {"query": src.get("query"), "items": src.get("docs")}
        if env is None:
            return None
        if not isinstance(env.get("items"), list) or not env["items"]:
            return None
        if "policy" not in env or not isinstance(env.get("policy"), dict):
            env["policy"] = {}
        return env
    except Exception:
        return None


def emit_retrieval_event(
    *,
    store: LocalStore,
    run_id: UUID,
    step: int,
    actor: str,
    namespace_label: str | None,
    thread_id: str | None,
    env: dict[str, Any],
    adapter_version: str,
    privacy_marks: dict[str, str] | None,
) -> tuple[Event, int]:
    try:
        items = env.get("items")
        if not isinstance(items, list) or not items:
            return (None, step)  # type: ignore[return-value]
        query = env.get("query")
        retriever = env.get("retriever")
        top_k = env.get("top_k")
        query_id = env.get("query_id")
        payload = {
            "query": query,
            "items": items,
            "policy": {"retriever": retriever, "top_k": top_k},
        }
        data_b = normalize_bytes(payload, privacy_marks=privacy_marks)
        blob = store.put_blob(run_id, step, BlobKind.MEMORY, data_b)

        from ..codec import to_bytes as _to_bytes

        hashes: dict[str, str] = {}
        try:
            if query is not None:
                hashes["query"] = hash_bytes(_to_bytes(query))
        except Exception:
            pass
        try:
            hashes["results"] = hash_bytes(_to_bytes({"items": items}))
        except Exception:
            pass
        labels: dict[str, str] = {}
        if namespace_label:
            labels["namespace"] = namespace_label
        if thread_id:
            labels["thread_id"] = thread_id
        if actor and actor != "graph":
            labels["node"] = actor
        try:
            labels["anchor_id"] = make_anchor_id(ActionType.RETRIEVAL, actor, labels)
        except Exception:
            pass
        ev = Event(
            run_id=run_id,
            step=step,
            action_type=ActionType.RETRIEVAL,
            actor=actor or "graph",
            output_ref=blob,
            hashes=hashes,
            labels=labels,
            model_meta={
                "adapter_version": adapter_version,
                "timewarp_version": _get_timewarp_version(),
                "framework": "langgraph",
                "mem_provider": "LangGraphState",
            },
            ts=tw_now(),
        )
        ev = ev.model_copy(
            update={
                "retriever": retriever if isinstance(retriever, str) else None,
                "top_k": int(top_k) if isinstance(top_k, int) else None,
                "query_id": str(query_id) if isinstance(query_id, str | int) else None,
            }
        )
        return ev, step + 1
    except Exception:
        return (None, step)  # type: ignore[return-value]
