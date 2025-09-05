from __future__ import annotations

import argparse
from typing import Any

from ...store import LocalStore
from ..helpers.jsonio import print_json


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    runs = list(store.list_runs(getattr(args, "project", None)))
    as_json = bool(getattr(args, "as_json", False))
    if as_json:
        rows: list[dict[str, Any]] = []
        for r in runs:
            try:
                events_count = store.count_events(r.run_id)
            except Exception:
                events_count = 0
            try:
                last_ts = store.last_event_ts(r.run_id)
                duration = (last_ts - r.started_at) if last_ts else None
            except Exception:
                duration = None
            branch_of = r.labels.get("branch_of") if r.labels else None
            rows.append(
                {
                    "run_id": str(r.run_id),
                    "project": r.project,
                    "name": r.name,
                    "started_at": r.started_at.isoformat(),
                    "duration": (str(duration) if duration else None),
                    "events": events_count,
                    "branch_of": branch_of,
                    "status": r.status,
                }
            )
        print_json(rows)
        return 0

    # Rich table mode (default)
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Timewarp Runs")
        table.add_column("Run ID", overflow="fold")
        table.add_column("Project/Name")
        table.add_column("Started")
        table.add_column("Duration")
        table.add_column("Events", justify="right")
        table.add_column("Branch Of")
        table.add_column("Status")
        for r in runs:
            proj_name = f"{r.project or ''}/{r.name or ''}"
            status = r.status or ""
            try:
                events_count = store.count_events(r.run_id)
            except Exception:
                events_count = 0
            try:
                last_ts = store.last_event_ts(r.run_id)
                duration = (last_ts - r.started_at) if last_ts else None
            except Exception:
                duration = None
            dur_text = str(duration) if duration else "-"
            branch_of = r.labels.get("branch_of") if r.labels else None
            table.add_row(
                str(r.run_id),
                proj_name,
                str(r.started_at),
                dur_text,
                str(events_count),
                branch_of or "",
                status,
            )
        console.print(table)
    except Exception:
        for r in runs:
            proj_name = f"{r.project or ''}/{r.name or ''}"
            status = r.status or ""
            try:
                events_count = store.count_events(r.run_id)
            except Exception:
                events_count = 0
            try:
                last_ts = store.last_event_ts(r.run_id)
                duration = (last_ts - r.started_at) if last_ts else None
            except Exception:
                duration = None
            dur_text = str(duration) if duration else "-"
            branch_of = r.labels.get("branch_of") if r.labels else ""
            base = f"{r.run_id}  {proj_name}  {r.started_at}"
            part2 = f"dur={dur_text}  events={events_count}  branch_of={branch_of}  {status}"
            print(f"{base}  {part2}")
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    lst = sub.add_parser("list")
    lst.add_argument("--project", dest="project", default=None, help="Filter by project")
    lst.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")
    lst.set_defaults(func=_handler)
