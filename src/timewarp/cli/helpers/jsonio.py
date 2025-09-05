from __future__ import annotations

from pathlib import Path
from typing import Any


def dumps_text(obj: Any) -> str:
    """Serialize to JSON text using orjson when available.

    Falls back to stdlib json with ensure_ascii=False.
    """
    try:
        from ...utils import json as _json_utils

        return _json_utils.dumps(obj).decode("utf-8")
    except Exception:
        import json as _json

        return _json.dumps(obj, ensure_ascii=False)


def print_json(obj: Any) -> None:
    print(dumps_text(obj))


def loads_file(path: Path) -> Any:
    """Read a JSON file as bytes and decode via orjson; fallback to json."""
    try:
        from ...utils import json as _json_utils

        return _json_utils.loads(path.read_bytes())
    except Exception:
        import json as _json

        return _json.loads(path.read_text(encoding="utf-8"))
