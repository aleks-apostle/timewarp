from __future__ import annotations

# Small helper to diff two runs and print minimal failing window.
#
# Usage:
#   uv run python -m examples.langgraph_demo.diff_runs            # use 2 latest runs
#   uv run python -m examples.langgraph_demo.diff_runs A B        # explicit run ids
from pathlib import Path
from typing import Any
from uuid import UUID

from timewarp.diff import bisect_divergence, first_divergence
from timewarp.store import LocalStore


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    store = LocalStore(db_path=Path("timewarp.sqlite3"), blobs_root=Path("blobs"))
    if len(args) >= 2:
        run_a = UUID(args[0])
        run_b = UUID(args[1])
    else:
        runs = store.list_runs()
        if len(runs) < 2:
            print("Need at least two runs recorded.")
            return 1
        run_a = runs[1].run_id  # older
        run_b = runs[0].run_id  # latest
    d1 = first_divergence(store, run_a, run_b, window=5)
    b = bisect_divergence(store, run_a, run_b, window=5)
    try:
        import orjson as _orjson

        payload: dict[str, Any] = {
            "run_a": str(run_a),
            "run_b": str(run_b),
            "first_divergence": (
                None
                if d1 is None
                else {
                    "step_a": d1.step_a,
                    "step_b": d1.step_b,
                    "reason": d1.reason,
                }
            ),
            "minimal_window": b,
        }
        print(_orjson.dumps(payload).decode("utf-8"))
    except Exception:
        import json as _json

        payload2: dict[str, Any] = {
            "run_a": str(run_a),
            "run_b": str(run_b),
            "first_divergence": (
                None
                if d1 is None
                else {
                    "step_a": d1.step_a,
                    "step_b": d1.step_b,
                    "reason": d1.reason,
                }
            ),
            "minimal_window": b,
        }
        print(_json.dumps(payload2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
