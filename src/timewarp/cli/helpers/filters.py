from __future__ import annotations


def parse_list_filters(tokens: list[str]) -> dict[str, str]:
    """Parse key=value tokens used by CLI event filters.

    Recognized keys: type, node, thread, namespace.
    """
    out: dict[str, str] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k in {"type", "node", "thread", "namespace"} and v:
            out[k] = v
    return out
