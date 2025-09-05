from __future__ import annotations

from collections.abc import AsyncIterable
from typing import Any, cast


def aiter_stream(
    graph: Any,
    inputs: dict[str, Any],
    config: dict[str, Any] | None,
    stream_kwargs: dict[str, Any],
) -> AsyncIterable[Any]:
    try:
        it = graph.astream(inputs, config or {}, **stream_kwargs)
        return cast(AsyncIterable[Any], it)
    except TypeError:
        stream_kwargs2 = dict(stream_kwargs)
        stream_kwargs2.pop("subgraphs", None)
        it2 = graph.astream(inputs, config or {}, **stream_kwargs2)
        return cast(AsyncIterable[Any], it2)
