from __future__ import annotations

import argparse
from typing import cast
from uuid import UUID

from ...events import ActionType, Event
from ...store import LocalStore
from ..helpers.blobs import read_json_blob as _read_json_blob
from ..helpers.jsonio import print_json


def _extract_tools_from_llm_event(
    store: LocalStore, e: Event
) -> tuple[list[object] | None, str | None]:
    tools_digest: str | None = None
    try:
        if e.hashes:
            hv = e.hashes.get("tools")
            if isinstance(hv, str):
                tools_digest = hv
    except Exception:
        pass
    if tools_digest is None:
        tools_digest = e.tools_digest
    tools_list: list[object] | None = None
    obj = _read_json_blob(store, e.input_ref)
    if isinstance(obj, dict):
        try:
            t = obj.get("tools")
            if isinstance(t, list):
                tools_list = t
        except Exception:
            tools_list = None
    return (tools_list, tools_digest)


def _collect_tools_called(
    events: list[Event], llm_index: int, thread_id: str | None, node: str | None
) -> list[Event]:
    out: list[Event] = []
    n = len(events)
    # scan forward until next LLM on the same thread
    for j in range(llm_index + 1, n):
        ev = events[j]
        if ev.labels.get("thread_id") == thread_id and ev.action_type is ActionType.LLM:
            break
        if ev.action_type is ActionType.TOOL:
            if thread_id is not None and ev.labels.get("thread_id") != thread_id:
                continue
            if node is not None and (ev.labels.get("node") or ev.actor) != node:
                continue
            out.append(ev)
    return out


def _build_tools_detail(store: LocalStore, events: list[Event], llm: Event) -> dict[str, object]:
    # llm index
    try:
        idx = next(i for i, x in enumerate(events) if x.step == llm.step)
    except StopIteration:
        idx = 0
    thread_id = llm.labels.get("thread_id")
    node = llm.labels.get("node") or (llm.actor if llm.actor != "graph" else None)
    tools_list, tools_digest = _extract_tools_from_llm_event(store, llm)
    called = _collect_tools_called(events, idx, thread_id, node)
    called_rows: list[dict[str, object]] = []
    for t in called:
        called_rows.append(
            {
                "step": t.step,
                "tool_kind": t.tool_kind or "",
                "tool_name": t.tool_name or "",
                "args_hash": (t.hashes.get("args") if t.hashes else None),
                "output_hash": (t.hashes.get("output") if t.hashes else None),
            }
        )
    out: dict[str, object] = {
        "llm_step": llm.step,
        "actor": llm.actor,
        "thread_id": thread_id or "",
        "tools_digest": tools_digest or "",
        "tools_count": (len(tools_list) if tools_list is not None else 0),
        "called": called_rows,
    }
    if tools_list is not None:
        out["tools"] = tools_list
    # include chunks_count when present
    try:
        if isinstance(llm.model_meta, dict) and "chunks_count" in llm.model_meta:
            cc_obj = llm.model_meta.get("chunks_count")
            if isinstance(cc_obj, int | str):
                out["chunks_count"] = int(cc_obj)
    except Exception:
        pass
    return out


def _print_tools_detail(detail: dict[str, object]) -> None:
    hdr = (
        f"LLM step: {detail.get('llm_step')}  actor={detail.get('actor')}  "
        f"thread={detail.get('thread_id')}"
    )
    print(hdr)
    td = detail.get("tools_digest") or ""
    print(f"tools_digest={td}")
    tc_obj = detail.get("tools_count") if "tools_count" in detail else None
    tc = int(tc_obj) if isinstance(tc_obj, int | str) else 0
    print(f"available_tools={tc}")
    if "tools" in detail and isinstance(detail["tools"], list):
        tools = cast(list[object], detail["tools"])
        head = tools[:10]
        for i, t in enumerate(head):
            print(f"  - tool[{i}]: {t}")
        if len(tools) > len(head):
            print(f"  ... ({len(tools) - len(head)} more)")
    called = detail.get("called")
    if isinstance(called, list):
        print("called tools:")
        for row in called:
            try:
                print(
                    f"  - step={row.get('step')} name={row.get('tool_name') or ''} "
                    f"kind={row.get('tool_kind') or ''} args_hash={row.get('args_hash') or ''}"
                )
            except Exception:
                print(f"  - {row}")


def _build_tools_summary(store: LocalStore, events: list[Event]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for e in events:
        if e.action_type is not ActionType.LLM:
            continue
        tools_list, tools_digest = _extract_tools_from_llm_event(store, e)
        try:
            idx = next(i for i, x in enumerate(events) if x.step == e.step)
        except StopIteration:
            idx = 0
        called = _collect_tools_called(
            events,
            idx,
            e.labels.get("thread_id"),
            e.labels.get("node") or (e.actor if e.actor != "graph" else None),
        )
        rows.append(
            {
                "step": e.step,
                "actor": e.actor,
                "thread_id": e.labels.get("thread_id") or "",
                "tools_digest": tools_digest or "",
                "available": len(tools_list) if tools_list is not None else 0,
                "called": len(called),
            }
        )
    return rows


def _print_tools_summary(rows: list[dict[str, object]]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Tools Summary (LLM steps)")
        table.add_column("Step", justify="right")
        table.add_column("Actor")
        table.add_column("Thread")
        table.add_column("Avail", justify="right")
        table.add_column("Called", justify="right")
        table.add_column("Digest")
        for r in rows:
            table.add_row(
                str(r.get("step")),
                str(r.get("actor") or ""),
                str(r.get("thread_id") or ""),
                str(r.get("available") or 0),
                str(r.get("called") or 0),
                str(r.get("tools_digest") or ""),
            )
        console.print(table)
    except Exception:
        for r in rows:
            line1 = (
                f"{r.get('step')!s:>4} [LLM] {r.get('actor')} "
                f"thr={r.get('thread_id')} avail={r.get('available')}"
            )
            line2 = f"called={r.get('called')} digest={r.get('tools_digest')}"
            print(line1)
            print("  " + line2)


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    events = store.list_events(UUID(args.run_id))
    # Detail view for a specific LLM step
    if getattr(args, "step", None) is not None:
        step = int(args.step)
        llm: Event | None = next(
            (e for e in events if e.step == step and e.action_type is ActionType.LLM), None
        )
        if llm is None:
            print("No LLM event at the specified step")
            return 1
        result = _build_tools_detail(store, events, llm)
        if bool(getattr(args, "as_json", False)):
            print_json(result)
            return 0
        _print_tools_detail(result)
        return 0

    # Summary across LLM steps
    rows = _build_tools_summary(store, events)
    if bool(getattr(args, "as_json", False)):
        print_json(rows)
        return 0
    _print_tools_summary(rows)
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    tlsp = sub.add_parser("tools", help="Show available tools (per LLM) and called tools")
    tlsp.add_argument("run_id", help="Run ID to inspect")
    tlsp.add_argument("--step", dest="step", type=int, default=None, help="LLM step to detail")
    tlsp.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")
    tlsp.set_defaults(func=_handler)
