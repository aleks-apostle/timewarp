from __future__ import annotations

from collections.abc import Callable
from typing import Any


def normalize_bytes(
    obj: Any,
    *,
    privacy_marks: dict[str, str] | None = None,
    extract_values_fn: Callable[[Any], Any | None] | None = None,
) -> bytes:
    from ..codec import to_bytes
    from ..events import redact

    redacted = obj
    try:
        redacted = redact(obj, privacy_marks or {})
    except Exception:
        pass
    try:
        return to_bytes(redacted)
    except Exception:
        # Fallbacks for non-JSON-serializable objects
        vals = None
        if extract_values_fn is not None:
            try:
                vals = extract_values_fn(redacted)
            except Exception:
                vals = None
        if vals is not None:
            return to_bytes(vals)
        try:
            if hasattr(redacted, "model_dump"):
                md = redacted.model_dump
                if callable(md):
                    return to_bytes(md(mode="json"))
        except Exception:
            pass
        return to_bytes({"_repr": repr(redacted)})


def extract_values(snapshot_like: Any) -> Any | None:
    try:
        if isinstance(snapshot_like, dict) and "values" in snapshot_like:
            vals = snapshot_like["values"]
            if isinstance(vals, dict | list):
                return vals
    except Exception:
        pass
    try:
        if hasattr(snapshot_like, "values"):
            vals2 = snapshot_like.values
            if isinstance(vals2, dict | list):
                return vals2
            if callable(vals2):
                out = vals2()
                if isinstance(out, dict | list):
                    return out
    except Exception:
        pass
    return None


def extract_next_nodes(values_like: Any) -> list[str] | None:
    try:
        src = values_like
        if isinstance(values_like, dict) and "values" in values_like:
            src = values_like["values"]
        if not isinstance(src, dict):
            return None
        nxt = src.get("next") or src.get("next_nodes")
        if isinstance(nxt, list) and all(isinstance(x, str) for x in nxt):
            return [str(x) for x in nxt]
    except Exception:
        return None
    return None


def extract_checkpoint_id(config: dict[str, Any] | None, values_like: Any | None) -> str | None:
    try:
        # Try from values payload first (shapes vary by runtime)
        if isinstance(values_like, dict):
            v = (
                values_like.get("values")
                if isinstance(values_like.get("values"), dict)
                else values_like
            )
            if isinstance(v, dict):
                cp = v.get("config")
                if isinstance(cp, dict):
                    conf2 = cp.get("configurable")
                    if isinstance(conf2, dict) and isinstance(
                        conf2.get("checkpoint_id"), str | int
                    ):
                        return str(conf2.get("checkpoint_id"))
    except Exception:
        pass
    try:
        if isinstance(config, dict):
            conf = config.get("configurable")
            if isinstance(conf, dict) and isinstance(conf.get("checkpoint_id"), str | int):
                return str(conf.get("checkpoint_id"))
    except Exception:
        pass
    return None


def serialize_messages_tuple(pair: Any) -> dict[str, Any]:
    msg = pair[0]
    meta = pair[1]

    def to_plain(x: Any) -> Any:
        try:
            if hasattr(x, "model_dump") and callable(x.model_dump):
                return x.model_dump(mode="json")
        except Exception:
            pass
        try:
            if hasattr(x, "content"):
                return {"content": x.content}
        except Exception:
            pass
        if isinstance(x, str | int | float | bool):
            return x
        if isinstance(x, dict):
            return x
        if isinstance(x, list | tuple):
            return [to_plain(y) for y in x]
        return {"_repr": repr(x)}

    return {"message": to_plain(msg), "metadata": to_plain(meta)}


def normalize_stream_item(
    update: Any, single_mode_label: str | None
) -> tuple[str | None, str | None, Any]:
    namespace_label: str | None = None
    stream_mode_label: str | None = None
    upd = update
    try:
        if (
            isinstance(update, tuple | list)
            and len(update) == 3
            and isinstance(update[0], tuple | list)
            and isinstance(update[1], str)
        ):
            ns = [str(x) for x in update[0]]
            namespace_label = "/".join(ns)
            stream_mode_label = update[1]
            upd = update[2]
            return namespace_label, stream_mode_label, upd
    except Exception:
        pass
    try:
        if isinstance(update, tuple | list) and len(update) == 2 and isinstance(update[0], str):
            stream_mode_label = update[0]
            upd = update[1]
            return namespace_label, stream_mode_label, upd
    except Exception:
        pass
    try:
        if (
            isinstance(update, tuple | list)
            and len(update) == 2
            and isinstance(update[0], tuple | list)
        ):
            ns = [str(x) for x in update[0]]
            namespace_label = "/".join(ns)
            upd = update[1]
            return namespace_label, stream_mode_label, upd
    except Exception:
        pass
    return namespace_label, single_mode_label, upd


def derive_actor_from_namespace(namespace_label: str, actor: str) -> str:
    try:
        last_seg = namespace_label.split("/")[-1]
        return last_seg.split(":")[0] if ":" in last_seg else last_seg
    except Exception:
        return actor
