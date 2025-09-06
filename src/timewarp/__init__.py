"""Timewarp core SDK.

Exposes core types and factories for event-sourced recording and deterministic replay.
"""

try:  # best-effort: avoid import errors for source checkouts
    from importlib.metadata import PackageNotFoundError as _PkgNotFound
    from importlib.metadata import version as _pkg_version

    try:
        __version__ = _pkg_version("timewarp-llm")
    except _PkgNotFound:  # pragma: no cover - occurs in editable/source contexts
        __version__ = "0+unknown"
except Exception:  # pragma: no cover - extremely defensive
    __version__ = "0+unknown"

from .codec import from_bytes, to_bytes, zstd_compress, zstd_decompress
from .determinism import SystemTimeProvider, TimeProvider, restore_rng, snapshot_rng
from .diff import make_anchor_key, realign_by_anchor
from .events import ActionType, BlobKind, BlobRef, Event, Run
from .langgraph import RecorderHandle, wrap
from .memory import rebuild_memory_snapshot
from .pruners import messages_pruner
from .replay import (
    AdapterInvariant,
    MissingBlob,
    Replay,
    ReplayError,
    ReplaySession,
    SchemaMismatch,
)

__all__ = [
    "ActionType",
    "AdapterInvariant",
    "BlobKind",
    "BlobRef",
    "Event",
    "MissingBlob",
    "RecorderHandle",
    "Replay",
    "ReplayError",
    "ReplaySession",
    "Run",
    "SchemaMismatch",
    "SystemTimeProvider",
    "TimeProvider",
    "__version__",
    "from_bytes",
    "make_anchor_key",
    "messages_pruner",
    "realign_by_anchor",
    "rebuild_memory_snapshot",
    "restore_rng",
    "snapshot_rng",
    "to_bytes",
    "wrap",
    "zstd_compress",
    "zstd_decompress",
]
