from __future__ import annotations

import argparse
from uuid import UUID

from ...events import Event
from ...store import LocalStore
from ..helpers.events import filter_events
from ..helpers.jsonio import print_json


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    events = store.list_events(UUID(args.run_id))
    filtered = filter_events(
        events,
        etype=(str(args.etype) if getattr(args, "etype", None) else None),
        node=(str(args.node) if getattr(args, "node", None) else None),
        thread=(str(args.thread_id) if getattr(args, "thread_id", None) else None),
        namespace=(str(args.namespace) if getattr(args, "namespace", None) else None),
        tool_kind=(str(args.tool_kind) if getattr(args, "tool_kind", None) else None),
        tool_name=(str(args.tool_name) if getattr(args, "tool_name", None) else None),
    )
    # paging
    off = max(0, int(getattr(args, "offset", 0)))
    lim = max(1, int(getattr(args, "limit", 1000000)))
    filtered = filtered[off : off + lim]
    if getattr(args, "as_json", False):
        rows = [
            {
                "step": e.step,
                "type": e.action_type.value,
                "actor": e.actor,
                "labels": e.labels,
            }
            for e in filtered
        ]
        print_json(rows)
        return 0

    def _compact_labels(e: Event) -> str:
        try:
            labels: list[str] = []
            sm = e.labels.get("stream_mode")
            ns = e.labels.get("namespace")
            tid = e.labels.get("thread_id")
            if sm:
                labels.append(f"sm={sm}")
            if ns:
                labels.append(f"ns={ns}")
            if tid:
                labels.append(f"thr={tid}")
            return ", ".join(labels)
        except Exception:
            return ""

    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"Events for {args.run_id}")
        table.add_column("Step", justify="right")
        table.add_column("Type")
        table.add_column("Actor")
        table.add_column("Labels")
        for e in filtered:
            table.add_row(str(e.step), e.action_type.value, e.actor, _compact_labels(e))
        console.print(table)
    except Exception:
        for e in filtered:
            print(
                f"{e.step:4d} {e.action_type.value:8s} {e.actor:10s} "
                f"{_compact_labels(e) or e.labels}"
            )
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    evp = sub.add_parser("events")
    evp.add_argument("run_id", help="Run ID to list events for")
    evp.add_argument("--type", dest="etype", default=None, help="Filter by action type")
    evp.add_argument("--node", dest="node", default=None, help="Filter by node/actor")
    evp.add_argument("--thread", dest="thread_id", default=None, help="Filter by thread id")
    evp.add_argument(
        "--namespace", dest="namespace", default=None, help="Filter by namespace label"
    )
    evp.add_argument(
        "--tool-kind", dest="tool_kind", default=None, help="Filter by tool_kind (e.g., MCP)"
    )
    evp.add_argument("--tool-name", dest="tool_name", default=None, help="Filter by tool_name")
    evp.add_argument("--offset", type=int, default=0, help="Pagination offset")
    evp.add_argument("--limit", type=int, default=1000000, help="Pagination limit")
    evp.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")
    evp.set_defaults(func=_handler)
