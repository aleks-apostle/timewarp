from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from ..determinism import now as tw_now
from ..events import ActionType, BlobKind, Event
from ..store import LocalStore
from .anchors import make_anchor_id
from .serialize import normalize_bytes
from .versioning import get_timewarp_version as _get_timewarp_version


def get_by_path(root: dict[str, Any], path: str) -> Any | None:
    try:
        cur: Any = root
        for seg in path.split("."):
            if not isinstance(cur, dict) or seg not in cur:
                return None
            cur = cur[seg]
        return cur
    except Exception:
        return None


def prune_mem_value(value: Any, pruner: Callable[[Any], Any] | None) -> Any:
    if pruner is None:
        return value
    try:
        pruned = pruner(value)
        return pruned if isinstance(pruned, dict | list | str | int | float | bool) else value
    except Exception:
        return value


def infer_mem_scope_from_path(path: str) -> str:
    p = path.lower()
    if "long" in p:
        return "long"
    if "short" in p:
        return "short"
    return "working"


@dataclass
class MemoryEmitter:
    store: LocalStore
    run_id: UUID
    adapter_version: str
    privacy_marks: dict[str, str] | None = None
    memory_pruner: Callable[[Any], Any] | None = None
    mem_prev: dict[str, str] = field(default_factory=dict, repr=False)

    def emit_from_values(
        self,
        *,
        step: int,
        actor: str,
        namespace_label: str | None,
        thread_id: str | None,
        values: dict[str, Any],
        memory_paths: tuple[str, ...] = (
            "messages",
            "history",
            "scratch",
            "artifacts",
            "memory",
        ),
        mem_space_resolver: Callable[[dict[str, str], str], str] | None = None,
    ) -> tuple[int, list[Event]]:
        events: list[Event] = []
        paths: tuple[str, ...] = tuple(memory_paths)
        for path in paths:
            v = get_by_path(values, path)
            if v is None:
                continue
            pruned = prune_mem_value(v, self.memory_pruner)
            payload = {"key": path, "value": pruned}
            data_b = normalize_bytes(payload, privacy_marks=self.privacy_marks)
            blob = self.store.put_blob(self.run_id, step, BlobKind.MEMORY, data_b)
            h = blob.sha256_hex
            # Prepare labels context early to compute mem_space
            labels: dict[str, str] = {}
            if namespace_label:
                labels["namespace"] = namespace_label
            if thread_id:
                labels["thread_id"] = thread_id
            if actor and actor != "graph":
                labels["node"] = actor
            labels["mem_op"] = "PUT"
            labels["mem_scope"] = infer_mem_scope_from_path(path)
            # Choose mem_space with resolver, defaulting to node/actor/graph
            mem_space = (
                mem_space_resolver(labels, actor)
                if mem_space_resolver is not None
                else (labels.get("node") or actor or "graph")
            )
            labels["mem_space"] = mem_space

            # Track prior hash per mem_space+path to avoid cross-agent suppression
            prev_key = f"{mem_space}:{path}"
            prev = self.mem_prev.get(prev_key)
            if prev is not None and prev == h:
                continue
            labels["mem_op"] = "PUT" if prev is None else "UPDATE"
            self.mem_prev[prev_key] = h
            try:
                labels["anchor_id"] = make_anchor_id(ActionType.MEMORY, actor, labels)
            except Exception:
                pass
            ev = Event(
                run_id=self.run_id,
                step=step,
                action_type=ActionType.MEMORY,
                actor=actor or "graph",
                output_ref=blob,
                hashes={"item": h},
                labels=labels,
                model_meta={
                    "adapter_version": self.adapter_version,
                    "timewarp_version": _get_timewarp_version(),
                    "framework": "langgraph",
                    "mem_provider": "LangGraphState",
                },
                ts=tw_now(),
            )
            ev = ev.model_copy(update={"mem_provider": "LangGraphState"})
            events.append(ev)
            step += 1
        return step, events
