from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, cast


def iter_stream_sync(
    graph: Any,
    inputs: dict[str, Any],
    config: dict[str, Any] | None,
    stream_kwargs: dict[str, Any],
) -> Iterable[Any]:
    try:
        it = graph.stream(inputs, config or {}, **stream_kwargs)
        return cast(Iterable[Any], it)
    except TypeError:
        stream_kwargs2 = dict(stream_kwargs)
        stream_kwargs2.pop("subgraphs", None)
        it2 = graph.stream(inputs, config or {}, **stream_kwargs2)
        return cast(Iterable[Any], it2)


def build_stream_kwargs(
    *,
    stream_modes: Sequence[str],
    stream_subgraphs: bool,
    durability: str | None,
    thread_id: str | None,
) -> dict[str, Any]:
    effective_durability = durability
    if effective_durability is None and thread_id:
        effective_durability = "sync"
    stream_kwargs: dict[str, Any] = {
        "stream_mode": list(stream_modes)
        if len(stream_modes) > 1
        else (stream_modes[0] if stream_modes else "updates"),
    }
    if stream_subgraphs:
        stream_kwargs["subgraphs"] = True
    if effective_durability is not None:
        stream_kwargs["durability"] = effective_durability
    return stream_kwargs
