from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict
from uuid import UUID

from ..bindings import try_pop_prompt_hash
from ..determinism import now as tw_now
from ..events import ActionType, BlobKind, Event, hash_bytes
from ..store import LocalStore
from .anchors import make_anchor_id
from .hashing import hash_prompt_ctx, hash_tools_list
from .serialize import normalize_bytes
from .versioning import get_timewarp_version as _get_timewarp_version


def finalize_messages_aggregate(
    *,
    store: LocalStore,
    run_id: UUID,
    step: int,
    agg_actor: str | None,
    agg_text: list[str],
    agg_chunks: list[dict[str, Any]],
    agg_labels: dict[str, str],
    privacy_marks: dict[str, str] | None,
    adapter_version: str,
) -> tuple[Event, int]:
    chunks_payload: dict[str, Any] = {"chunks": agg_chunks}
    chunks_b = normalize_bytes(chunks_payload, privacy_marks=privacy_marks)
    chunks_ref = store.put_blob(run_id, step, BlobKind.STATE, chunks_b)

    payload: dict[str, Any] = {
        "message": {"content": "".join(agg_text)},
        "metadata": {"chunks_count": len(agg_chunks)},
        "chunks_ref": chunks_ref.model_dump(mode="json"),
    }
    payload_b2 = normalize_bytes(payload, privacy_marks=privacy_marks)
    out_blob2 = store.put_blob(run_id, step, BlobKind.OUTPUT, payload_b2)

    labels2 = dict(agg_labels)
    labels2["stream_mode"] = "messages"
    if agg_actor:
        labels2.setdefault("node", agg_actor)

    prompt_hash: str | None = None
    provider: str | None = None
    model: str | None = None
    params_meta: dict[str, Any] = {}
    tools_list: list[Any] = []
    ctx_messages: Any | None = None
    _prompt_hash_failed = False
    try:
        sources: list[Any] = []
        for ch in agg_chunks:
            meta = ch.get("metadata") if isinstance(ch, dict) else None
            if not isinstance(meta, dict):
                continue
            for key in ("llm_input_messages", "input_messages", "messages", "prompt"):
                if key in meta:
                    sources.append(meta[key])
                    if ctx_messages is None:
                        ctx_messages = meta[key]
            if provider is None and isinstance(meta.get("provider"), str):
                provider = str(meta.get("provider"))
            if model is None and isinstance(meta.get("model"), str):
                model = str(meta.get("model"))
            for p in ("temperature", "top_p", "tool_choice"):
                val = meta.get(p)
                if val is None and isinstance(meta.get("params"), dict):
                    params = meta["params"]
                    try:
                        val = params.get(p)
                    except Exception:
                        val = None
                if val is not None and p not in params_meta:
                    if isinstance(val, str | int | float | bool):
                        params_meta[p] = val
            try:
                t = meta.get("tools") if isinstance(meta, dict) else None
                if t is None and isinstance(meta, dict):
                    t = meta.get("available_tools")
                if isinstance(t, list) and t:
                    tools_list.extend(t)
            except Exception:
                pass
        if sources:
            # Keep aggregated semantics by hashing the combined sources payload deterministically
            from ..codec import to_bytes as _to_bytes

            prompt_hash = hash_bytes(_to_bytes({"sources": sources}))
    except Exception:
        prompt_hash = None
        _prompt_hash_failed = True

    try:
        anchor_id2 = make_anchor_id(ActionType.LLM, agg_actor or "graph", labels2)
        if anchor_id2:
            labels2["anchor_id"] = anchor_id2
    except Exception:
        pass

    if prompt_hash:
        labels2["hash_source"] = "aggregated"

    ev_hashes: dict[str, str] = {"output": out_blob2.sha256_hex}
    if prompt_hash:
        ev_hashes["prompt"] = prompt_hash
    ev_tools_digest: str | None = None
    input_ref = None
    try:
        if tools_list:
            ev_tools_digest = hash_tools_list(tools_list)
            ev_hashes["tools"] = ev_tools_digest
            if ctx_messages is not None:
                ctx_obj = {"messages": ctx_messages, "tools": tools_list}
                parts_b = normalize_bytes(ctx_obj, privacy_marks=privacy_marks)
                input_ref = store.put_blob(run_id, step, BlobKind.INPUT, parts_b)
                ev_hashes["prompt_ctx"] = hash_prompt_ctx(messages=ctx_messages, tools=tools_list)
    except Exception:
        pass

    ev = Event(
        run_id=run_id,
        step=step,
        action_type=ActionType.LLM,
        actor=agg_actor or "graph",
        input_ref=input_ref,
        output_ref=out_blob2,
        hashes=ev_hashes,
        labels=labels2,
        model_meta={
            "adapter_version": adapter_version,
            "timewarp_version": _get_timewarp_version(),
            "framework": "langgraph",
            "chunks_count": len(agg_chunks),
            "prompt_hash_agg_failed": True if _prompt_hash_failed else False,
            **({"provider": provider} if provider else {}),
            **({"model": model} if model else {}),
            **params_meta,
        },
        ts=tw_now(),
    )
    if ev_tools_digest is not None:
        ev = ev.model_copy(update={"tools_digest": ev_tools_digest})
    try:
        if "prompt" not in ev.hashes:
            staged = try_pop_prompt_hash()
            if staged:
                new_labels = dict(ev.labels or {})
                new_labels["hash_source"] = "staged"
                ev = ev.model_copy(
                    update={"hashes": {**ev.hashes, "prompt": staged}, "labels": new_labels}
                )
    except Exception:
        pass
    return ev, step + 1


class NormalizedMessageChunk(TypedDict, total=False):
    message: Any
    metadata: dict[str, Any]


@dataclass
class MessagesAggregator:
    store: LocalStore
    run_id: UUID
    adapter_version: str
    privacy_marks: dict[str, str] | None

    agg_key: tuple[str, str | None, str | None] | None = None
    agg_chunks: list[NormalizedMessageChunk] = field(default_factory=list)
    agg_text: list[str] = field(default_factory=list)
    agg_labels: dict[str, str] = field(default_factory=dict)
    agg_actor: str | None = None

    def start(
        self,
        *,
        actor: str,
        namespace_label: str | None,
        thread_id: str | None,
        normalized: dict[str, Any],
    ) -> None:
        self.agg_key = (actor, namespace_label, thread_id)
        self.agg_labels = {}
        if namespace_label:
            self.agg_labels["namespace"] = namespace_label
        if thread_id:
            self.agg_labels["thread_id"] = thread_id
        try:
            meta = normalized.get("metadata")
            if isinstance(meta, dict):
                ln = meta.get("langgraph_node")
                if ln:
                    self.agg_labels["node"] = str(ln)
        except Exception:
            pass
        self.agg_actor = actor

    def append(self, normalized: dict[str, Any]) -> None:
        self.agg_chunks.append(normalized)  # type: ignore[arg-type]
        try:
            msg = normalized.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                self.agg_text.append(msg["content"])
            elif isinstance(msg, str):
                self.agg_text.append(msg)
        except Exception:
            pass

    def key(self) -> tuple[str, str | None, str | None] | None:
        return self.agg_key

    def has_pending(self) -> bool:
        return bool(self.agg_key)

    def reset(self) -> None:
        self.agg_key = None
        self.agg_chunks = []
        self.agg_text = []
        self.agg_labels = {}
        self.agg_actor = None

    def flush(self, *, step: int) -> tuple[Event, int]:
        ev, step2 = finalize_messages_aggregate(
            store=self.store,
            run_id=self.run_id,
            step=step,
            agg_actor=self.agg_actor,
            agg_text=self.agg_text,
            agg_chunks=self.agg_chunks,  # type: ignore[arg-type]
            agg_labels=self.agg_labels,
            privacy_marks=self.privacy_marks,
            adapter_version=self.adapter_version,
        )
        self.reset()
        return ev, step2
