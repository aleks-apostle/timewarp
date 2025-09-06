from __future__ import annotations

from typing import Any, TypedDict
from uuid import UUID

from ..codec import from_bytes
from ..events import ActionType, BlobRef
from ..memory import rebuild_memory_snapshot
from ..store import LocalStore


class DSPyExample(TypedDict, total=False):
    inputs: dict[str, Any]
    memory: dict[str, Any]
    output: Any
    meta: dict[str, Any]


def _read_json_blob(store: LocalStore, ref: BlobRef | None) -> Any | None:
    if ref is None:
        return None
    try:
        return from_bytes(store.get_blob(ref))
    except Exception:
        return None


def _extract_messages_from_llm_event(
    store: LocalStore, out_payload: Any, in_payload: Any
) -> Any | None:
    """Best-effort extraction of input messages for a given LLM event.

    Priority:
    1) messages from input_ref payload (when messages aggregation was available)
    2) llm_input_messages directly present in the output payload (common in demos)
    3) messages embedded in chunks_ref->chunks[*].metadata
       (messages | input_messages | llm_input_messages)
    """

    # 1) messages via input payload from messages aggregator
    try:
        if isinstance(in_payload, dict) and isinstance(in_payload.get("messages"), list):
            return in_payload.get("messages")
    except Exception:
        pass

    # 2) llm_input_messages included directly in the output payload (updates stream)
    try:
        if isinstance(out_payload, dict):
            for key in ("llm_input_messages", "input_messages", "messages"):
                val = out_payload.get(key)
                if isinstance(val, list):
                    return val
    except Exception:
        pass

    # 3) Follow chunks_ref when present (messages aggregator)
    try:
        if isinstance(out_payload, dict) and isinstance(out_payload.get("chunks_ref"), dict):
            # Validate a BlobRef-like dict and load it
            # We import here to avoid circular imports during type checking
            ref_dict = out_payload.get("chunks_ref")
            ref = BlobRef.model_validate(ref_dict)
            chunks_obj = _read_json_blob(store, ref)
            if isinstance(chunks_obj, dict) and isinstance(chunks_obj.get("chunks"), list):
                for ch in chunks_obj.get("chunks", []):
                    meta = None
                    try:
                        meta = ch.get("metadata") if isinstance(ch, dict) else None
                    except Exception:
                        meta = None
                    if isinstance(meta, dict):
                        for key in ("llm_input_messages", "input_messages", "messages", "prompt"):
                            if key in meta and isinstance(meta[key], list | str):
                                return meta[key]
    except Exception:
        pass

    return None


def _extract_output_value(out_payload: Any) -> Any:
    """Best-effort selection of an output value for DSPy label.

    Prefers a text-like assistant message when present, otherwise returns the full payload.
    """
    try:
        if isinstance(out_payload, dict):
            msg = out_payload.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str | int | float):
                return msg.get("content")
            # Some providers may use 'text'
            if isinstance(out_payload.get("text"), str | int | float):
                return out_payload.get("text")
    except Exception:
        pass
    return out_payload


def build_dspy_dataset(
    store: LocalStore, run_id: UUID, agents: list[str] | None = None
) -> dict[str, list[DSPyExample]]:
    """Build a per-agent dataset suitable for DSPy optimizers.

    For every LLM event in the run, this reconstructs the memory snapshot at step-1
    for the corresponding thread and assembles an example with:
      - inputs: {"messages": [...]} when available (best-effort)
      - memory: per-agent memory view from the snapshot
      - output: recorded model output (string when available, else JSON payload)
      - meta: {"step": int, "thread": str | None, "agent": str}

    Returns a mapping agent -> list[examples]. Examples are JSON-serializable.
    """
    events = store.list_events(run_id)
    by_agent: dict[str, list[DSPyExample]] = {}

    def want_agent(name: str) -> bool:
        if agents is None:
            return True
        return name in agents

    for e in events:
        if e.action_type is not ActionType.LLM:
            continue
        agent = e.labels.get("node") or (e.actor if e.actor else "graph")
        agent = str(agent)
        if not want_agent(agent):
            continue

        step = int(e.step)
        thread_id = e.labels.get("thread_id")

        # Memory snapshot at step-1 narrowed to the same thread when available
        snap = rebuild_memory_snapshot(store, run_id, step - 1, thread_id=thread_id)
        # Select the agent-specific view when present
        mem_view: dict[str, Any] = {}
        try:
            by_space = snap.get("by_space") if isinstance(snap, dict) else None
            if (
                isinstance(by_space, dict)
                and agent in by_space
                and isinstance(by_space[agent], dict)
            ):
                mem_view = by_space[agent]
        except Exception:
            mem_view = {}

        # Gather I/O payloads
        in_obj = _read_json_blob(store, e.input_ref)
        out_obj = _read_json_blob(store, e.output_ref)

        messages = _extract_messages_from_llm_event(store, out_obj, in_obj)
        inputs: dict[str, Any] = {}
        if messages is not None:
            inputs["messages"] = messages

        example: DSPyExample = {
            "inputs": inputs,
            "memory": mem_view,
            "output": _extract_output_value(out_obj),
            "meta": {"step": step, "thread": thread_id, "agent": agent},
        }

        by_agent.setdefault(agent, []).append(example)

    return by_agent
