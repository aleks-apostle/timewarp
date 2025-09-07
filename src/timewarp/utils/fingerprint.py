from __future__ import annotations

import platform
from importlib import metadata
from typing import Final

_DEPS: Final[tuple[str, ...]] = (
    "langgraph",
    "langchain_core",
    "zstandard",
    "orjson",
)


def _get_version_safe(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except Exception:
        return None


def runtime_labels() -> dict[str, str]:
    """Return compact runtime fingerprint labels for a Run.

    Keys:
    - fp.py: python version
    - fp.os: platform triplet
    - fp.<dep>: version for selected deps when import metadata is available
    """
    labels: dict[str, str] = {}
    try:
        labels["fp.py"] = platform.python_version()
    except Exception:
        pass
    try:
        labels["fp.os"] = platform.platform()
    except Exception:
        pass
    for d in _DEPS:
        v = _get_version_safe(d)
        if v:
            labels[f"fp.{d}"] = v
    return labels
