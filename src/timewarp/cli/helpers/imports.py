from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import cast


def load_factory(spec: str) -> Callable[[], object]:
    """Import a factory callable from a "module:function" spec.

    Raises ValueError with a clear message on malformed specs or missing attributes.
    """
    if ":" not in spec:
        raise ValueError("Factory must be in 'module:function' format")
    mod_name, func_name = spec.split(":", 1)
    try:
        mod = import_module(mod_name)
    except Exception as exc:  # pragma: no cover - import errors
        raise ValueError(f"Failed to import module '{mod_name}': {exc}") from exc
    try:
        factory = getattr(mod, func_name)
    except Exception as exc:  # pragma: no cover - attribute missing
        raise ValueError(f"Module '{mod_name}' has no attribute '{func_name}'") from exc
    if not callable(factory):
        raise ValueError(f"Attribute '{func_name}' is not callable")
    return cast(Callable[[], object], factory)
