from __future__ import annotations

from pathlib import Path
from typing import Any

from ...events import Event
from ...store import LocalStore


def format_state_pretty(
    obj: Any, *, max_str: int = 200, max_items: int = 50, indent: int = 2
) -> str:
    """Pretty-format JSON-like state with truncation and size hints.

    Uses the core JSON codec for normalization to preserve determinism across runs.
    """

    def _trunc(v: Any) -> Any:
        if isinstance(v, str):
            if len(v) > max_str:
                return v[:max_str] + f"... <truncated {len(v) - max_str} chars>"
            return v
        if isinstance(v, list):
            head = v[:max_items]
            tail = len(v) - len(head)
            lst_out = [_trunc(x) for x in head]
            if tail > 0:
                lst_out.append(f"<... {tail} more items>")
            return lst_out
        if isinstance(v, dict):
            dict_out: dict[str, Any] = {}
            for k, val in v.items():
                dict_out[str(k)] = _trunc(val)
            return dict_out
        return v

    try:
        from ...codec import to_bytes

        normalized = _trunc(obj)
        return to_bytes(normalized).decode("utf-8")
    except Exception:
        return repr(obj)


def dump_event_output_to_file(store: LocalStore, e: Event, path: Path) -> None:
    """Write the event's output JSON to a file.

    Prefers output_ref; falls back to input_ref if output is missing.
    Raises on IO or decode errors.
    """
    ref = e.output_ref or e.input_ref
    if not ref:
        raise ValueError("event has no output or input blob")
    raw = store.get_blob(ref)
    from ...codec import from_bytes, to_bytes

    obj = from_bytes(raw)
    data = to_bytes(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
