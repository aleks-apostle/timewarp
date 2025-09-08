from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class NetworkBlocked(RuntimeError):
    """Raised when a network operation is attempted while networking is disabled."""


@contextmanager
def no_network() -> Iterator[None]:
    """Context manager that disables outbound network access within the process.

    Patches common sync HTTP/TCP entry points to raise NetworkBlocked. Best-effort and
    dependency-free; only patches modules that are available in the environment.
    """

    patches: list[tuple[Any, str, Any]] = []

    def _patch(obj: Any, name: str, replacement: Any) -> None:
        if obj is None or not hasattr(obj, name):
            return
        orig = getattr(obj, name)
        patches.append((obj, name, orig))
        setattr(obj, name, replacement)

    def _raise(*_args: Any, **_kwargs: Any) -> Any:
        raise NetworkBlocked("network egress disabled during replay")

    try:
        import socket as _socket

        # Block top-level helpers
        _patch(_socket, "create_connection", _raise)
        _patch(_socket, "getaddrinfo", _raise)

        # Block instance methods on the socket class
        _patch(_socket.socket, "connect", _raise)
        _patch(_socket.socket, "connect_ex", _raise)
    except Exception:
        pass

    try:
        import http.client as _httpc

        # Block HTTP(S) connection establishment
        _patch(_httpc.HTTPConnection, "connect", _raise)
        _patch(_httpc.HTTPSConnection, "connect", _raise)
        _patch(_httpc.HTTPConnection, "request", _raise)
        _patch(_httpc.HTTPSConnection, "request", _raise)
    except Exception:
        pass

    try:  # pragma: no cover - optional
        import urllib.request as _urllib

        _patch(_urllib, "urlopen", _raise)
    except Exception:
        pass

    try:  # pragma: no cover - optional
        import importlib as _importlib

        _requests = _importlib.import_module("requests")
        _patch(_requests.sessions.Session, "request", _raise)
    except Exception:
        pass

    try:  # pragma: no cover - optional
        import importlib as _importlib

        _aiohttp = _importlib.import_module("aiohttp")
        _patch(_aiohttp.ClientSession, "_request", _raise)
    except Exception:
        pass

    try:
        yield None
    finally:
        # Restore all patched attributes in reverse order
        for obj, name, orig in reversed(patches):
            try:
                setattr(obj, name, orig)
            except Exception:
                # Best-effort restore
                continue


__all__ = ["NetworkBlocked", "no_network"]
