from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast
from uuid import UUID

from ...diff import first_divergence
from ...events import ActionType, BlobRef, Event, redact
from ...replay import Replay, ReplayError
from ...store import LocalStore
from ...utils.diffing import struct_diff as _struct_diff
from ..helpers.blobs import read_json_blob as _read_json_blob
from ..helpers.filters import parse_list_filters
from ..helpers.state import dump_event_output_to_file, format_state_pretty


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    rep = Replay(store=store, run_id=UUID(args.run_id))
    # Print basic run/version context
    evs = list(rep.iter_timeline())
    schema_v = evs[0].schema_version if evs else "-"
    adapter_versions: set[str] = set()
    framework: str | None = None
    for e in evs:
        mm = e.model_meta or {}
        if isinstance(mm, dict):
            av = mm.get("adapter_version")
            if isinstance(av, str):
                adapter_versions.add(av)
        if framework is None:
            try:
                if isinstance(mm, dict) and isinstance(mm.get("framework"), str):
                    framework = mm.get("framework")
            except Exception:
                pass
    if framework is None:
        # Fallback to run metadata when available
        try:
            for r in store.list_runs():
                if r.run_id == UUID(args.run_id):
                    if isinstance(r.framework, str):
                        framework = r.framework
                    break
        except Exception:
            framework = None
    adapters = ",".join(sorted(adapter_versions))
    fw = framework or ""
    print(f"schema={schema_v} adapter_versions={adapters} framework={fw}")
    # Interactive REPL
    print(
        "Commands: list [type=.. node=.. thread=.. namespace=..] | show N | tokens N |\n"
        "blob N [input|output|state] | goto N | step | next [TYPE] | inject N <json> |\n"
        "skip N | firstdiv RUN_B | state [--pretty] | savepatch STEP FILE | lastllm |\n"
        "memory | memory show N | memory diff A B [key=dot.path] | prompt N |\n"
        "tools [N] | help | quit"
    )
    while True:
        try:
            line = input("> ").strip()
        except (KeyboardInterrupt, EOFError):  # pragma: no cover - interactive
            print()
            return 0
        if not line:
            continue
        if line in ("q", "quit", "exit"):
            return 0
        if line == "list":
            _print_timeline(rep)
            continue
        if line.startswith("list "):
            filters = parse_list_filters(line.split()[1:])
            all_events = list(rep.iter_timeline())
            # Map to types for filtering
            from ..helpers.events import filter_events as _filter

            filtered = _filter(
                all_events,
                etype=filters.get("type"),
                node=filters.get("node"),
                thread=filters.get("thread"),
                namespace=filters.get("namespace"),
            )
            _print_timeline_filtered(filtered)
            continue
        if line.startswith("listp"):
            # Paged printing to avoid loading entire run in memory
            _print_timeline_paged(store, UUID(args.run_id), page_size=2000)
            continue
        if line.startswith("show "):
            _, s = line.split(maxsplit=1)
            step = int(s)
            evt: Event | None = next((e for e in rep.iter_timeline() if e.step == step), None)
            if not evt:
                print("No such step")
            else:
                _print_event(evt, store)
            continue
        if line.strip() == "memory":
            try:
                _repl_memory_list(rep)
            except Exception as exc:
                print("<memory list failed:", exc, ">")
            continue
        if line.startswith("memory show "):
            parts = line.split()
            if len(parts) < 3:
                print("Usage: memory show STEP")
                continue
            try:
                step = int(parts[2])
            except Exception:
                print("Usage: memory show STEP")
                continue
            try:
                _repl_memory_show(rep, store, step)
            except Exception as exc:
                print("<memory show failed:", exc, ">")
            continue
        if line.startswith("memory diff "):
            parts = line.split()
            if len(parts) < 4:
                print("Usage: memory diff A B [key=dot.path]")
                continue
            try:
                a_step = int(parts[2])
                b_step = int(parts[3])
            except Exception:
                print("Usage: memory diff A B [key=dot.path]")
                continue
            key = None
            for tok in parts[4:]:
                if tok.startswith("key="):
                    key = tok.split("=", 1)[1]
                    break
            try:
                _repl_memory_diff(rep, store, a_step, b_step, key)
            except Exception as exc:
                print("<memory diff failed:", exc, ">")
            continue
        if line.startswith("prompt "):
            parts = line.split()
            if len(parts) != 2:
                print("Usage: prompt STEP")
                continue
            try:
                step = int(parts[1])
            except Exception:
                print("Usage: prompt STEP")
                continue
            try:
                _repl_prompt(rep, store, step)
            except Exception as exc:
                print("<prompt failed:", exc, ">")
            continue
        if line.strip() == "tools" or line.startswith("tools "):
            parts = line.split()
            step2: int | None = None
            if len(parts) == 2:
                try:
                    step2 = int(parts[1])
                except Exception:
                    print("Usage: tools [STEP]")
                    continue
            try:
                _repl_tools(rep, store, step2)
            except Exception as exc:
                print("<tools failed:", exc, ">")
            continue
        if line.startswith("tokens "):
            try:
                _, s = line.split(maxsplit=1)
                step = int(s)
            except Exception:
                print("Usage: tokens N")
                continue
            evt2: Event | None = next((e for e in rep.iter_timeline() if e.step == step), None)
            if not evt2 or not evt2.output_ref:
                print("No such step or no output blob")
                continue
            _print_tokens(evt2, store)
            continue
        if line.startswith("goto "):
            _, s = line.split(maxsplit=1)
            rep.goto(int(s))
            print("pos=", rep._pos)
            continue
        if line == "step":
            rep.step()
            print("pos=", rep._pos)
            continue
        if line.startswith("inject "):
            try:
                _, s, payload_text = line.split(maxsplit=2)
                rep.inject(int(s), __import__("json").loads(payload_text))
                print("Injected at step", s)
            except Exception as e:  # pragma: no cover
                print("Inject failed:", e)
            continue
        if line.startswith("skip "):
            try:
                _, s = line.split(maxsplit=1)
                rep.skip(int(s))
                print("Skipped step", s)
            except Exception as e:  # pragma: no cover
                print("Skip failed:", e)
            continue
        if line.startswith("next"):
            parts = line.split()
            t = parts[1] if len(parts) > 1 else None
            at = None
            if t:
                try:
                    at = ActionType[t]
                except Exception:
                    print("Unknown type; use LLM|TOOL|DECISION|HITL|SNAPSHOT|SYS")
                    continue
            rep.next(at)
            print("pos=", rep._pos)
            continue
        if line.startswith("blob "):
            parts = line.split()
            if len(parts) < 2:
                print("Usage: blob STEP [input|output|state]")
                continue
            step = int(parts[1])
            which = parts[2] if len(parts) > 2 else None
            evt2 = next((e for e in rep.iter_timeline() if e.step == step), None)
            if not evt2:
                print("No such step")
                continue
            ref: BlobRef | None
            if which == "input":
                ref = evt2.input_ref
            elif which == "state":
                ref = evt2.output_ref if evt2.action_type.name == "SNAPSHOT" else None
            else:
                ref = evt2.output_ref or evt2.input_ref
            if not ref:
                print("No blob for this step")
                continue
            raw = store.get_blob(ref)
            print(raw.decode("utf-8", errors="replace"))
            continue
        if line.startswith(("firstdiv ", "diff ")):
            try:
                _, run_b = line.split(maxsplit=1)
                d = first_divergence(store, UUID(args.run_id), UUID(run_b))
                if not d:
                    print("No divergence")
                else:
                    rep.goto(d.step_a)
                    print(f"First divergence at step {d.step_a}: {d.reason}")
                    if d.diff_struct:
                        print("STRUCT DIFF:")
                        print(d.diff_struct)
                    if d.diff_text:
                        print("TEXT DIFF:")
                        print(d.diff_text)
            except Exception as e:  # pragma: no cover
                print("firstdiv failed:", e)
            continue
        if line == "lastllm":
            _print_last_llm(rep, store)
            continue
        if line == "help":
            cmds = (
                "Commands: list [type=.. node=.. thread=.. namespace=..] | show N | tokens N | "
                "blob N [input|output|state] | goto N | step | next [TYPE] | inject N <json> | "
                "skip N | firstdiv RUN_B | state [--pretty] | savepatch STEP FILE | lastllm | "
                "memory | memory show N | memory diff A B "
                "[key=dot.path] | prompt N | tools [N] | help | quit"
            )
            print(cmds)
            continue
        if line.startswith("state"):
            try:
                pretty = False
                parts = line.split()
                if len(parts) > 1 and parts[1] in ("--pretty", "pretty"):
                    pretty = True
                obj = rep.inspect_state()
                if pretty:
                    txt = format_state_pretty(obj)
                    print(txt)
                else:
                    print(obj)
            except ReplayError as e:  # pragma: no cover
                print("Replay error:", e)
            continue
        if line.startswith("savepatch "):
            parts = line.split()
            if len(parts) != 3:
                print("Usage: savepatch STEP FILE")
                continue
            try:
                step = int(parts[1])
            except Exception:
                print("STEP must be an integer")
                continue
            path = parts[2]
            evt3: Event | None = next((e for e in rep.iter_timeline() if e.step == step), None)
            if not evt3:
                print("No such step")
                continue
            try:
                dump_event_output_to_file(store, evt3, Path(path))
                print(f"Wrote patch to {path}")
            except Exception as ex:  # pragma: no cover
                print("savepatch failed:", ex)
            continue
        if line.startswith("copypatch "):
            parts = line.split()
            if len(parts) < 2 or len(parts) > 3:
                print("Usage: copypatch STEP [FILE]")
                continue
            try:
                step = int(parts[1])
            except Exception:
                print("STEP must be an integer")
                continue
            out_path = Path(parts[2]) if len(parts) == 3 else Path("patches") / f"alt_{step}.json"
            evt4: Event | None = next((e for e in rep.iter_timeline() if e.step == step), None)
            if not evt4:
                print("No such step")
                continue
            try:
                dump_event_output_to_file(store, evt4, out_path)
                print(f"Copied patch to {out_path}")
            except Exception as ex:
                print("copypatch failed:", ex)
            continue
        if line == "snapshot":
            try:
                ev = rep.snapshot_now()
                print(f"Snapshot appended at step {ev.step}")
            except ReplayError as e:  # pragma: no cover
                print("Snapshot failed:", e)
            continue
        print("Unknown command")
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    dbg = sub.add_parser("debug")
    dbg.add_argument("run_id", help="Run ID")
    dbg.set_defaults(func=_handler)


def _print_timeline(rep: Replay) -> None:
    events = list(rep.iter_timeline())
    _print_timeline_filtered(events)


def _print_timeline_filtered(events: list[Event]) -> None:
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Timeline")
        table.add_column("Step", justify="right")
        table.add_column("Type")
        table.add_column("Actor")
        table.add_column("Labels")
        for e in events:
            badge = _badge(e.action_type.value)
            labels = []
            sm = e.labels.get("stream_mode")
            ns = e.labels.get("namespace")
            tid = e.labels.get("thread_id")
            if sm:
                labels.append(f"sm={sm}")
            if ns:
                labels.append(f"ns={ns}")
            if tid:
                labels.append(f"thr={tid}")
            table.add_row(str(e.step), str(badge), e.actor, ", ".join(labels))
        console.print(table)
    except Exception:
        for e in events:
            print(f"{e.step:4d} {_plain_badge(e.action_type.value)} {e.actor:10s} {e.labels}")


def _print_timeline_paged(store: LocalStore, run_id: UUID, page_size: int = 1000) -> None:
    off = 0
    total_printed = 0
    while True:
        batch = store.list_events_window(run_id, off, page_size)
        if not batch:
            break
        for e in batch:
            print(f"{e.step:4d} {_plain_badge(e.action_type.value)} {e.actor:10s} {e.labels}")
            total_printed += 1
        off += len(batch)


## format_state_pretty and dump_event_output_to_file centralized in cli.helpers.state


def _print_event(e: Event, store: LocalStore) -> None:
    print(f"Step {e.step}  Type={e.action_type.value}  Actor={e.actor}")
    try:
        if isinstance(e.model_meta, dict):
            prov = e.model_meta.get("provider")
            mod = e.model_meta.get("model")
            info = "/".join(
                [x for x in [str(prov) if prov else None, str(mod) if mod else None] if x]
            )
            if info:
                print(f"Model: {info}")
    except Exception:
        pass
    if e.labels:
        print("Labels:", e.labels)
    ref = e.output_ref or e.input_ref
    if ref is None:
        return
    try:
        raw = store.get_blob(ref)
        text = raw.decode("utf-8", errors="replace")
        if len(text) > 500:
            text = text[:500] + "..."
        print("Preview:\n", text)
    except Exception:
        print("<blob read failed>")


def _badge(kind: str) -> object:
    try:
        from rich.text import Text

        styles = {
            "LLM": "bold cyan",
            "TOOL": "bold magenta",
            "HITL": "bold yellow",
            "SNAPSHOT": "bold green",
            "SYS": "dim",
            "DECISION": "bold blue",
            "ERROR": "bold red",
        }
        style = styles.get(kind, "")
        return Text(kind, style=style)
    except Exception:
        return kind


def _plain_badge(kind: str) -> str:
    symbols = {
        "LLM": "[LLM]",
        "TOOL": "[TOOL]",
        "HITL": "[HITL]",
        "SNAPSHOT": "[SNAP]",
        "SYS": "[SYS]",
        "DECISION": "[DEC]",
        "ERROR": "[ERR]",
    }
    return symbols.get(kind, kind)


def _print_last_llm(rep: Replay, store: LocalStore) -> None:
    pos = rep._pos if rep._pos > 0 else 0
    events = list(rep.iter_timeline())
    idx = min(pos - 1, len(events) - 1)
    while idx >= 0:
        e = events[idx]
        if e.action_type is ActionType.LLM and e.output_ref:
            print(f"Last LLM at step {e.step}")
            _print_event(e, store)
            return
        idx -= 1
    print("No LLM event found before current position")


def _print_tokens(e: Event, store: LocalStore) -> None:
    if not e.output_ref:
        print("No output blob for this event")
        return
    try:
        from ...codec import from_bytes

        payload = from_bytes(store.get_blob(e.output_ref))
    except Exception:
        print("<failed to read output blob>")
        return
    if not isinstance(payload, dict):
        print("<output is not a JSON object>")
        return
    messages = payload.get("messages") if isinstance(payload, dict) else None
    tools = payload.get("tools") if isinstance(payload, dict) else None
    est_tokens = _estimate_tokens(messages, tools)
    print(f"estimated_tokens≈{est_tokens}")


def _repl_memory_list(rep: Replay) -> None:
    events = list(rep.iter_timeline())
    mem = [e for e in events if e.action_type in (ActionType.MEMORY, ActionType.RETRIEVAL)]
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="Memory/Retrieval Events")
        table.add_column("Step", justify="right")
        table.add_column("Type")
        table.add_column("Actor")
        table.add_column("Thread")
        for e in mem:
            table.add_row(str(e.step), e.action_type.value, e.actor, e.labels.get("thread_id", ""))
        console.print(table)
    except Exception:
        for e in mem:
            thr = e.labels.get("thread_id", "")
            print(f"{e.step:4d} {e.action_type.value:8s} {e.actor:10s} thr={thr}")


def _repl_memory_show(rep: Replay, store: LocalStore, step: int) -> None:
    e = next((x for x in rep.iter_timeline() if x.step == step), None)
    if not e or e.action_type not in (ActionType.MEMORY, ActionType.RETRIEVAL):
        print("No MEMORY/RETRIEVAL event at that step")
        return
    obj = _read_json_blob(store, e.output_ref)
    if obj is None:
        print("<no blob>")
        return
    try:
        obj = redact(obj, e.privacy_marks or {})
    except Exception:
        pass
    txt = format_state_pretty(obj)
    print(txt)


def _pluck_path(obj: object, path: str | None) -> object:
    if not path or not isinstance(obj, dict | list):
        return obj
    cur: object = obj
    try:
        for part in path.split("."):
            if isinstance(cur, dict):
                cur = cast(object, cur.get(part))
            elif isinstance(cur, list):
                idx = int(part)
                cur = cur[idx]
            else:
                return cur
    except Exception:
        return cur
    return cur


def _repl_memory_diff(
    rep: Replay, store: LocalStore, a_step: int, b_step: int, key_path: str | None
) -> None:
    ea = next((x for x in rep.iter_timeline() if x.step == a_step), None)
    eb = next((x for x in rep.iter_timeline() if x.step == b_step), None)
    if not ea or not eb:
        print("Steps not found")
        return
    if ea.action_type not in (ActionType.MEMORY, ActionType.RETRIEVAL) or eb.action_type not in (
        ActionType.MEMORY,
        ActionType.RETRIEVAL,
    ):
        print("Both steps must be MEMORY/RETRIEVAL events")
        return
    oa = _read_json_blob(store, ea.output_ref)
    ob = _read_json_blob(store, eb.output_ref)
    if oa is None or ob is None:
        print("Missing blobs for diff")
        return
    try:
        oa = redact(oa, ea.privacy_marks or {})
        ob = redact(ob, eb.privacy_marks or {})
    except Exception:
        pass
    oa2 = _pluck_path(oa, key_path)
    ob2 = _pluck_path(ob, key_path)
    try:
        dd = _struct_diff(oa2, ob2)
        print(dd)
    except Exception:
        sa = format_state_pretty(oa2)
        sb = format_state_pretty(ob2)
        print(sa)
        print("--- vs ---")
        print(sb)


def _estimate_tokens(messages: object | None, tools: object | None) -> int:
    total_chars = 0
    try:
        if messages is not None:
            from ...codec import to_bytes as _to_bytes

            total_chars += len(_to_bytes(messages))
    except Exception:
        pass
    try:
        if tools is not None:
            from ...codec import to_bytes as _to_bytes

            total_chars += len(_to_bytes(tools))
    except Exception:
        pass
    return max(0, total_chars // 4)


def _repl_prompt(rep: Replay, store: LocalStore, step: int) -> None:
    e = next((x for x in rep.iter_timeline() if x.step == step), None)
    if not e or e.action_type is not ActionType.LLM:
        print("No LLM event at that step")
        return
    print(f"LLM step {e.step}  actor={e.actor}")
    try:
        if e.hashes:
            pr = e.hashes.get("prompt")
            td = e.hashes.get("tools") or e.tools_digest
            pc = e.hashes.get("prompt_ctx")
            print(f"hashes: prompt={pr or ''} tools={td or ''} prompt_ctx={pc or ''}")
    except Exception:
        pass
    obj = _read_json_blob(store, e.input_ref)
    messages = None
    tools = None
    if isinstance(obj, dict):
        messages = obj.get("messages")
        tools = obj.get("tools")
    try:
        if isinstance(messages, list):
            print(f"messages_count={len(messages)}")
            head = messages[:5]
            for i, m in enumerate(head):
                if isinstance(m, dict) and isinstance(m.get("content"), str):
                    print(f"  {i:02d}: {m['content'][:120]}")
                else:
                    print(f"  {i:02d}: {m}")
            if len(messages) > len(head):
                print(f"  ... ({len(messages) - len(head)} more)")
    except Exception:
        pass
    try:
        if isinstance(tools, list):
            print(f"tools_count={len(tools)}")
            t_head = tools[:10]
            for i, t in enumerate(t_head):
                print(f"  - tool[{i}]: {t}")
            if len(tools) > len(t_head):
                print(f"  ... ({len(tools) - len(t_head)} more)")
    except Exception:
        pass
    est_tokens = _estimate_tokens(messages, tools)
    print(f"estimated_tokens≈{est_tokens}")


def _repl_tools(rep: Replay, store: LocalStore, step: int | None) -> None:
    # Reuse the tools command's helpers to avoid dupes
    from . import tools as _tools

    events = list(rep.iter_timeline())
    if step is None:
        rows = _tools._build_tools_summary(store, events)
        _tools._print_tools_summary(rows)
        return
    e = next((x for x in events if x.step == step and x.action_type is ActionType.LLM), None)
    if not e:
        print("No LLM event at that step")
        return
    detail = _tools._build_tools_detail(store, events, e)
    _tools._print_tools_detail(detail)
