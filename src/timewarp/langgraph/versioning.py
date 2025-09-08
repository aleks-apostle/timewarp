from __future__ import annotations


def get_timewarp_version() -> str:
    """Return the installed timewarp package version, or a fallback string."""
    try:
        from timewarp import __version__ as _tw_version

        return _tw_version
    except Exception:
        return "0+unknown"


def lib_versions_meta() -> dict[str, str]:
    """Best-effort library versions for provenance in model_meta."""
    out: dict[str, str] = {}
    try:
        import importlib

        lg = importlib.import_module("langgraph")
        v = getattr(lg, "__version__", None)
        if isinstance(v, str):
            out["langgraph_version"] = v
    except Exception:
        pass
    try:
        import importlib

        lcc = importlib.import_module("langchain_core")
        v = getattr(lcc, "__version__", None)
        if isinstance(v, str):
            out["langchain_core_version"] = v
    except Exception:
        pass
    return out
