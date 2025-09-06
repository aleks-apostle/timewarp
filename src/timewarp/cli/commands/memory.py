from __future__ import annotations

import argparse
from typing import Any
from uuid import UUID

from ...memory import rebuild_memory_snapshot
from ...store import LocalStore
from ...utils.diffing import struct_diff
from ..helpers.jsonio import print_json


def _print_summary(snapshot: dict[str, Any]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Memory Summary (per space)")
        table.add_column("Space")
        table.add_column("Short", justify="right")
        table.add_column("Working", justify="right")
        table.add_column("Long", justify="right")
        table.add_column("Retrievals", justify="right")
        by_space = snapshot.get("by_space", {}) if isinstance(snapshot, dict) else {}
        for space, v in (by_space or {}).items():
            try:
                short_n = len(v.get("short") or {})
                work_n = len(v.get("working") or {})
                long_n = len(v.get("long") or {})
                ret_n = len(v.get("retrievals") or [])
                table.add_row(str(space), str(short_n), str(work_n), str(long_n), str(ret_n))
            except Exception:
                continue
        console.print(table)
    except Exception:
        by_space = snapshot.get("by_space", {}) if isinstance(snapshot, dict) else {}
        for space, v in (by_space or {}).items():
            short_n = len(v.get("short") or {})
            work_n = len(v.get("working") or {})
            long_n = len(v.get("long") or {})
            ret_n = len(v.get("retrievals") or [])
            print(f"{space}: short={short_n} working={work_n} long={long_n} retrievals={ret_n}")


def _print_show(snapshot: dict[str, Any], space: str | None) -> None:
    by_space = snapshot.get("by_space", {}) if isinstance(snapshot, dict) else {}
    if space:
        v = by_space.get(space)
        print_json({space: v})
        return
    print_json(by_space)


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    run_id = UUID(args.run_id)
    mode = getattr(args, "mode", "summary")
    thread = getattr(args, "thread_id", None)
    if mode == "summary":
        step = int(args.step)
        snap = rebuild_memory_snapshot(store, run_id, step, thread_id=thread)
        if bool(getattr(args, "as_json", False)):
            print_json(snap)
            return 0
        _print_summary(snap)
        return 0
    if mode == "show":
        step = int(args.step)
        snap = rebuild_memory_snapshot(store, run_id, step, thread_id=thread)
        if bool(getattr(args, "as_json", False)):
            print_json(snap)
            return 0
        _print_show(snap, getattr(args, "space", None))
        return 0
    if mode == "diff":
        step_a = int(args.step_a)
        step_b = int(args.step_b)
        thread = getattr(args, "thread_id", None)
        space = getattr(args, "space", None)
        scope = getattr(args, "scope", None)
        key = getattr(args, "key", None)
        snap_a = rebuild_memory_snapshot(store, run_id, step_a, thread_id=thread)
        snap_b = rebuild_memory_snapshot(store, run_id, step_b, thread_id=thread)

        def select(obj: dict[str, Any]) -> Any:
            if isinstance(obj, dict):
                cur: Any = obj.get("by_space", {})
            else:
                cur = {}
            if space:
                cur = cur.get(space, {}) if isinstance(cur, dict) else {}
            if scope:
                cur = cur.get(scope, {}) if isinstance(cur, dict) else {}
            if key and isinstance(cur, dict):
                # dotted path lookup
                tmp: Any = cur
                for seg in str(key).split("."):
                    if isinstance(tmp, dict) and seg in tmp:
                        tmp = tmp[seg]
                    else:
                        tmp = None
                        break
                cur = tmp
            return cur

        a_obj = select(snap_a)
        b_obj = select(snap_b)
        diff_obj = struct_diff(a_obj, b_obj)
        if bool(getattr(args, "as_json", False)):
            print_json(diff_obj)
        else:
            print(diff_obj)
        # Non-empty diff may be interpreted by callers; return 0 always
        return 0
    print("Unknown memory mode")
    return 1


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    mem = sub.add_parser("memory", help="Inspect reconstructed memory snapshots")
    mem_sub = mem.add_subparsers(dest="mode", required=True)

    sm = mem_sub.add_parser("summary", help="Summary of memory by space at a step")
    sm.add_argument("run_id", help="Run ID")
    sm.add_argument("--step", dest="step", type=int, required=True, help="Step upper bound")
    sm.add_argument("--thread", dest="thread_id", default=None, help="Thread ID filter")
    sm.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")
    sm.set_defaults(func=_handler)

    sh = mem_sub.add_parser("show", help="Show snapshot content by space at a step")
    sh.add_argument("run_id", help="Run ID")
    sh.add_argument("--step", dest="step", type=int, required=True, help="Step upper bound")
    sh.add_argument("--space", dest="space", default=None, help="Memory space (agent)")
    sh.add_argument("--thread", dest="thread_id", default=None, help="Thread ID filter")
    sh.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")
    sh.set_defaults(func=_handler)

    df = mem_sub.add_parser("diff", help="Structural diff between two snapshots")
    df.add_argument("run_id", help="Run ID")
    df.add_argument("step_a", type=int, help="First step")
    df.add_argument("step_b", type=int, help="Second step")
    df.add_argument("--space", dest="space", default=None, help="Memory space (agent)")
    df.add_argument(
        "--scope", dest="scope", default=None, help="Scope within space: short|working|long"
    )
    df.add_argument(
        "--key",
        dest="key",
        default=None,
        help="Optional dotted key inside the selected object",
    )
    df.add_argument("--thread", dest="thread_id", default=None, help="Thread ID filter")
    df.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")
    df.set_defaults(func=_handler)
