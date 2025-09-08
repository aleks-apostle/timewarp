from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from ..store import LocalStore

Handler = Callable[[argparse.Namespace, LocalStore], int]


def _register_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    # Import locally to avoid import-time side effects and optional deps.
    # Debug REPL is not yet modularized; skip if unavailable.
    from .commands import debug as _cmd_debug
    from .commands import diff as _cmd_diff
    from .commands import dspy as _cmd_dspy
    from .commands import events as _cmd_events
    from .commands import export as _cmd_export
    from .commands import fsck as _cmd_fsck
    from .commands import inject as _cmd_inject
    from .commands import list_runs as _cmd_list
    from .commands import memory as _cmd_memory
    from .commands import resume as _cmd_resume
    from .commands import tools as _cmd_tools

    _cmd_list.register(sub)
    _cmd_events.register(sub)
    _cmd_fsck.register(sub)
    _cmd_tools.register(sub)
    _cmd_diff.register(sub)
    _cmd_export.register(sub)
    _cmd_resume.register(sub)
    _cmd_inject.register(sub)
    _cmd_memory.register(sub)
    _cmd_dspy.register(sub)
    _cmd_debug.register(sub)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="timewarp")
    # Positional args are optional with sane defaults; flags override when provided
    p.add_argument(
        "db", nargs="?", default=None, help="Path to SQLite DB (default: timewarp.sqlite3)"
    )
    p.add_argument("blobs", nargs="?", default=None, help="Path to blobs root (default: ./blobs)")
    p.add_argument("--db", dest="db_file", default=None, help="Path to SQLite DB file")
    p.add_argument("--blobs", dest="blobs_dir", default=None, help="Path to blobs directory")
    sub = p.add_subparsers(dest="cmd", required=True)
    _register_commands(sub)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    db_path = Path(args.db_file or args.db or "timewarp.sqlite3")
    blobs_root = Path(args.blobs_dir or args.blobs or "blobs")
    store = LocalStore(db_path=db_path, blobs_root=blobs_root)

    func: Handler | None = getattr(args, "func", None)
    if func is None:
        # Should not happen due to required=True
        return 1
    return int(func(args, store))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
