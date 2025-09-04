from __future__ import annotations

import argparse
from pathlib import Path
from uuid import UUID

from .diff import bisect_divergence, first_divergence
from .events import ActionType, BlobRef, Event
from .replay import Replay, ReplayError
from .store import LocalStore


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="timewarp")
    p.add_argument("db", help="Path to SQLite DB")
    p.add_argument("blobs", help="Path to blobs root")

    sub = p.add_subparsers(dest="cmd", required=True)
    lst = sub.add_parser("list")
    lst.add_argument("--project", dest="project", default=None, help="Filter by project")
    lst.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")

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
    evp.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")

    dbg = sub.add_parser("debug")
    dbg.add_argument("run_id", help="Run ID")

    res = sub.add_parser(
        "resume",
        help="Resume a run deterministically using recorded outputs",
        description="Resume a run from a prior checkpoint using playback wrappers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Programmatic API: from timewarp import Replay\n"
            'session = Replay.resume(store, app_factory="mod:make", run_id=<UUID>,\n'
            '    from_step=<int|None>, thread_id="t-1", strict_meta=False, freeze_time=False)\n'
        ),
    )
    res.add_argument("run_id", help="Run ID")
    res.add_argument("--from", dest="from_step", type=int, default=None, help="Step to resume from")
    res.add_argument("--thread", dest="thread_id", default=None, help="Thread ID for LangGraph")
    res.add_argument(
        "--app",
        dest="app_factory",
        required=True,
        help="Python path to factory returning compiled graph: module:function",
    )
    res.add_argument(
        "--strict-meta",
        dest="strict_meta",
        action="store_true",
        help="Enforce model_meta validation in replay (provider/model/params)",
    )
    res.add_argument(
        "--freeze-time",
        dest="freeze_time",
        action="store_true",
        help="Freeze time during replay to recorded event timestamps",
    )

    inj = sub.add_parser("inject")
    inj.add_argument("run_id", help="Run ID")
    inj.add_argument("step", type=int, help="Step to override (ignored with --state-patch)")
    inj.add_argument("--output", dest="output_file", help="JSON file with replacement output")
    inj.add_argument(
        "--state-patch",
        dest="state_patch_file",
        help="JSON file with state patch to apply at latest checkpoint",
    )
    inj.add_argument("--thread", dest="thread_id", default=None, help="Thread ID for LangGraph")
    inj.add_argument(
        "--app",
        dest="app_factory",
        required=True,
        help="Python path to factory returning compiled graph: module:function",
    )
    inj.add_argument(
        "--strict-meta",
        dest="strict_meta",
        action="store_true",
        help="Enforce model_meta validation in replay (provider/model/params)",
    )
    inj.add_argument(
        "--record-fork",
        dest="record_fork",
        action="store_true",
        help="Execute the fork immediately and record a new run",
    )
    inj.add_argument(
        "--freeze-time",
        dest="freeze_time",
        action="store_true",
        help="Freeze time during replay to recorded event timestamps",
    )

    ddf = sub.add_parser("diff")
    ddf.add_argument("run_a")
    ddf.add_argument("run_b")
    ddf.add_argument("--window", type=int, default=5, help="Anchor realignment window (default 5)")
    ddf.add_argument("--json", dest="as_json", action="store_true", help="Emit JSON output")
    ddf.add_argument(
        "--bisect", dest="use_bisect", action="store_true", help="Find minimal failing window"
    )
    ddf.add_argument(
        "--fail-on-divergence",
        dest="fail_on_divergence",
        action="store_true",
        help="Exit with non-zero status when a divergence is found",
    )

    # export subcommands
    exp = sub.add_parser("export")
    exp_sub = exp.add_subparsers(dest="exporter", required=True)
    exp_ls = exp_sub.add_parser("langsmith")
    exp_ls.add_argument("run_id", help="Run ID to export")
    exp_ls.add_argument(
        "--include-blobs",
        dest="include_blobs",
        action="store_true",
        help="Inline small blobs as JSON where possible",
    )

    args = p.parse_args(argv)
    store = LocalStore(db_path=Path(args.db), blobs_root=Path(args.blobs))

    if args.cmd == "list":
        runs = list(store.list_runs(getattr(args, "project", None)))
        # JSON mode
        try:
            as_json = bool(getattr(args, "as_json", False))
        except Exception:
            as_json = False
        if as_json:
            rows: list[dict[str, object]] = []
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
            try:
                import orjson as _orjson

                print(_orjson.dumps(rows).decode("utf-8"))
            except Exception:
                import json as _json

                print(_json.dumps(rows, ensure_ascii=False))
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
                # Compute insights
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

    if args.cmd == "events":
        events = store.list_events(UUID(args.run_id))
        filtered = _filter_events(
            events,
            etype=(str(args.etype) if getattr(args, "etype", None) else None),
            node=(str(args.node) if getattr(args, "node", None) else None),
            thread=(str(args.thread_id) if getattr(args, "thread_id", None) else None),
            namespace=(str(args.namespace) if getattr(args, "namespace", None) else None),
            tool_kind=(str(args.tool_kind) if getattr(args, "tool_kind", None) else None),
            tool_name=(str(args.tool_name) if getattr(args, "tool_name", None) else None),
        )
        if getattr(args, "as_json", False):
            try:
                import orjson as _orjson

                rows = [
                    {
                        "step": e.step,
                        "type": e.action_type.value,
                        "actor": e.actor,
                        "labels": e.labels,
                    }
                    for e in filtered
                ]
                print(_orjson.dumps(rows).decode("utf-8"))
            except Exception:
                import json as _json

                rows2 = [
                    {
                        "step": e.step,
                        "type": e.action_type.value,
                        "actor": e.actor,
                        "labels": e.labels,
                    }
                    for e in filtered
                ]
                print(_json.dumps(rows2, ensure_ascii=False))
            return 0
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
                table.add_row(str(e.step), e.action_type.value, e.actor, ", ".join(e.labels.keys()))
            console.print(table)
        except Exception:
            for e in filtered:
                print(f"{e.step:4d} {e.action_type.value:8s} {e.actor:10s} {e.labels}")
        return 0

    if args.cmd == "diff":
        use_bisect = bool(getattr(args, "use_bisect", False))
        if use_bisect:
            b = bisect_divergence(store, UUID(args.run_a), UUID(args.run_b), window=args.window)
            if getattr(args, "as_json", False):
                exit_code = 0
                try:
                    from typing import Any, cast

                    import orjson as _orjson

                    payload: object
                    if b is None:
                        payload = {"result": None}
                    else:
                        payload = cast(dict[str, Any], b)
                        try:
                            if hasattr(args, "fail_on_divergence") and bool(
                                args.fail_on_divergence
                            ):
                                exit_code = 1
                        except Exception:
                            pass
                    print(_orjson.dumps(payload).decode("utf-8"))
                except Exception:
                    import json as _json
                    from typing import Any, cast

                    payload2: object
                    if b is None:
                        payload2 = {"result": None}
                    else:
                        payload2 = cast(dict[str, Any], b)
                        try:
                            if hasattr(args, "fail_on_divergence") and bool(
                                args.fail_on_divergence
                            ):
                                exit_code = 1
                        except Exception:
                            pass
                    print(_json.dumps(payload2, ensure_ascii=False))
                return exit_code
            if b is None:
                print("No divergence: runs equivalent by step/order/hashes")
                return 0
            print(
                f"Minimal failing window: A[{b['start_a']}:{b['end_a']}], "
                f"B[{b['start_b']}:{b['end_b']}] — cause={b['cause']}"
            )
            try:
                if hasattr(args, "fail_on_divergence") and bool(args.fail_on_divergence):
                    return 1
            except Exception:
                pass
            return 0

        d = first_divergence(store, UUID(args.run_a), UUID(args.run_b), window=args.window)
        if getattr(args, "as_json", False):
            exit_code = 0
            try:
                import orjson as _orjson

                diff_payload: dict[str, object]
                if d is None:
                    diff_payload = {"result": None}
                else:
                    diff_payload = {
                        "step_a": d.step_a,
                        "step_b": d.step_b,
                        "reason": d.reason,
                        "diff_struct": d.diff_struct,
                        "diff_text": d.diff_text,
                    }
                    try:
                        if hasattr(args, "fail_on_divergence") and bool(args.fail_on_divergence):
                            exit_code = 1
                    except Exception:
                        pass
                print(_orjson.dumps(diff_payload).decode("utf-8"))
            except Exception:
                import json as _json

                payload_json: dict[str, object]
                if d is None:
                    payload_json = {"result": None}
                else:
                    payload_json = {
                        "step_a": d.step_a,
                        "step_b": d.step_b,
                        "reason": d.reason,
                        "diff_struct": d.diff_struct,
                        "diff_text": d.diff_text,
                    }
                    try:
                        if hasattr(args, "fail_on_divergence") and bool(args.fail_on_divergence):
                            exit_code = 1
                    except Exception:
                        pass
                print(_json.dumps(payload_json, ensure_ascii=False))
            return exit_code
        if d is None:
            print("No divergence: runs equivalent by step/order/hashes")
            return 0
        print(f"First divergence at A:{d.step_a} B:{d.step_b}: {d.reason}")
        if d.diff_struct:
            print("STRUCT DIFF:")
            print(d.diff_struct)
        if d.diff_text:
            print("TEXT DIFF:")
            print(d.diff_text)
        # Support CI usage: fail when divergence present and flag is set
        try:
            if hasattr(args, "fail_on_divergence") and bool(args.fail_on_divergence):
                return 1
        except Exception:
            pass
        return 0

    if args.cmd == "export":
        if args.exporter == "langsmith":
            try:
                from uuid import UUID as _UUID

                import orjson as _orjson

                from timewarp.exporters.langsmith import serialize_run as _serialize_run
            except Exception as exc:  # pragma: no cover - dependency/import errors
                print("Export failed: missing dependencies:", exc)
                return 1
            payload = _serialize_run(
                store, _UUID(args.run_id), include_blobs=bool(getattr(args, "include_blobs", False))
            )
            try:
                print(_orjson.dumps(payload).decode("utf-8"))
            except Exception:
                import json as _json

                print(_json.dumps(payload, ensure_ascii=False))
            return 0
        print("Unknown exporter:", args.exporter)
        return 1

    if args.cmd == "resume":
        # Lazy import to avoid optional deps at CLI parse time
        from importlib import import_module

        from timewarp.adapters import installers as _installers
        from timewarp.replay import LangGraphReplayer

        try:
            mod_name, func_name = args.app_factory.split(":", 1)
            mod = import_module(mod_name)
            from collections.abc import Callable
            from typing import Any, cast

            factory = cast(Callable[[], Any], getattr(mod, func_name))
            graph = factory()
        except Exception as exc:
            print("Failed to import app factory:", exc)
            return 1

        from timewarp.replay import PlaybackLLM, PlaybackMemory, PlaybackTool  # typing

        def installer_resume(llm: PlaybackLLM, tool: PlaybackTool, memory: PlaybackMemory) -> None:
            try:
                # Toggle strict model_meta validation when requested
                try:
                    llm.strict_meta = bool(args.strict_meta)
                    tool.strict_meta = bool(args.strict_meta)
                except Exception:
                    pass
                _installers.bind_langgraph_playback(graph, llm, tool, memory)
            except Exception as exc:  # pragma: no cover
                print("Warning: failed to bind playback wrappers:", exc)

        replayer = LangGraphReplayer(graph=graph, store=store)
        session = replayer.resume(
            UUID(args.run_id),
            args.from_step,
            args.thread_id,
            install_wrappers=installer_resume,
            freeze_time=bool(getattr(args, "freeze_time", False)),
        )
        print("Resumed run:", args.run_id)
        print("checkpoint_id=", session.checkpoint_id)
        # Show summarized result if small
        try:
            import orjson as _orjson

            blob = _orjson.dumps(session.result)
            txt = blob.decode("utf-8")
            print("result:")
            print(txt[:2000])
        except Exception:
            print("result:", session.result)
        return 0

    if args.cmd == "inject":
        from importlib import import_module

        from timewarp.adapters import installers as _installers
        from timewarp.replay import LangGraphReplayer

        # Validate mutually exclusive modes
        if not args.output_file and not args.state_patch_file:
            print("Provide either --output <file> or --state-patch <file>")
            return 1
        if args.output_file and args.state_patch_file:
            print("Use only one of --output or --state-patch")
            return 1
        replacement = None
        patch_obj = None
        if args.output_file:
            try:
                import orjson as _orjson

                replacement = _orjson.loads(Path(args.output_file).read_bytes())
            except Exception as exc:
                print("Failed to read replacement output:", exc)
                return 1
        if args.state_patch_file:
            try:
                import orjson as _orjson

                patch_obj = _orjson.loads(Path(args.state_patch_file).read_bytes())
            except Exception as exc:
                print("Failed to read state patch:", exc)
                return 1
        try:
            mod_name, func_name = args.app_factory.split(":", 1)
            mod = import_module(mod_name)
            from collections.abc import Callable
            from typing import Any, cast

            factory = cast(Callable[[], Any], getattr(mod, func_name))
            graph = factory()
        except Exception as exc:
            print("Failed to import app factory:", exc)
            return 1

        from collections.abc import Callable

        from timewarp.replay import (  # local import for typing
            PlaybackLLM,
            PlaybackMemory,
            PlaybackTool,
        )

        teardowns: list[Callable[[], None]] = []

        def installer_inject(llm: PlaybackLLM, tool: PlaybackTool, memory: PlaybackMemory) -> None:
            try:
                try:
                    llm.strict_meta = bool(args.strict_meta)
                    tool.strict_meta = bool(args.strict_meta)
                except Exception:
                    pass
                td = _installers.bind_langgraph_playback(graph, llm, tool, memory)
                # retain teardown in outer scope
                teardowns.append(td)
            except Exception as exc:  # pragma: no cover
                print("Warning: failed to bind playback wrappers:", exc)

        # State patch mode
        if patch_obj is not None:
            try:
                from typing import Any

                cfg: dict[str, object] = {"configurable": {}}
                if args.thread_id is not None:
                    cfg = {"configurable": {"thread_id": args.thread_id}}
                get_state = getattr(graph, "get_state", None)
                if not callable(get_state):
                    print("Graph does not support get_state; cannot apply state patch")
                    return 1
                snap: Any = get_state(cfg)
                inner_cfg: Any | None = None
                cfg_attr = getattr(snap, "config", None)
                if cfg_attr is not None:
                    inner_cfg = cfg_attr
                elif isinstance(snap, dict):
                    inner_cfg = snap.get("config")
                if inner_cfg is None:
                    print("Could not extract config from state snapshot; aborting state patch")
                    return 1
                update_state = getattr(graph, "update_state", None)
                if not callable(update_state):
                    print("Graph does not support update_state; cannot apply state patch")
                    return 1
                new_cfg: Any = update_state(inner_cfg, values=patch_obj)
                new_cp = None
                try:
                    new_cp = new_cfg["configurable"]["checkpoint_id"]
                except Exception:
                    try:
                        if hasattr(new_cfg, "get"):
                            conf = new_cfg.get("configurable")
                            if isinstance(conf, dict):
                                new_cp = conf.get("checkpoint_id")
                    except Exception:
                        new_cp = None
                print("Applied state patch; new checkpoint_id=", new_cp)
                return 0
            except Exception as exc:
                print("Failed to apply state patch:", exc)
                return 1

        # Output override mode
        replayer = LangGraphReplayer(graph=graph, store=store)
        new_id = replayer.fork_with_injection(
            UUID(args.run_id),
            args.step,
            replacement,
            args.thread_id,
            install_wrappers=installer_inject,
            freeze_time=bool(getattr(args, "freeze_time", False)),
        )
        # If recording now, execute the graph with a recorder bound to the new run id
        if bool(getattr(args, "record_fork", False)):
            try:
                # Retrieve original input payload
                evs = store.list_events(UUID(args.run_id))
                orig_input = None
                for ev in evs:
                    if ev.input_ref is not None:
                        from timewarp.codec import from_bytes as _from_bytes

                        orig_input = _from_bytes(store.get_blob(ev.input_ref))
                        break
                if orig_input is None:
                    print("Could not locate original input for the run; aborting fork recording")
                    return 1
                # Compose new Run metadata: branch_of + override_step
                from timewarp.events import Run as _Run

                # Try to copy project/name from original run metadata when available
                proj = None
                name = None
                for r in store.list_runs():
                    if r.run_id == UUID(args.run_id):
                        proj = r.project
                        name = r.name
                        break
                new_run = _Run(
                    run_id=new_id,
                    project=proj,
                    name=name,
                    framework="langgraph",
                    labels={"branch_of": str(args.run_id), "override_step": str(args.step)},
                )
                from timewarp.adapters.langgraph import LangGraphRecorder as _LGRecorder

                rec = _LGRecorder(
                    graph=graph,
                    store=store,
                    run=new_run,
                    snapshot_every=20,
                    stream_modes=("updates", "messages", "values"),
                    stream_subgraphs=True,
                )
                cfg2: dict[str, object] = {"configurable": {}}
                if args.thread_id is not None:
                    cfg2 = {"configurable": {"thread_id": args.thread_id}}
                _ = rec.invoke(orig_input, config=cfg2)
                print("Fork executed and recorded:", new_id)
                return 0
            finally:
                # Teardown playback patches if installed
                for td in teardowns:
                    try:
                        td()
                    except Exception:
                        pass
        else:
            print("Fork prepared with override at step", args.step)
            print("New run id:", new_id)
            print(
                "Note: to record the fork immediately, re-run with --record-fork or "
                "run your app with the recorder."
            )
            return 0

    if args.cmd == "debug":
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
                fw = mm.get("framework")
                if isinstance(av, str):
                    adapter_versions.add(av)
                if framework is None and isinstance(fw, str):
                    framework = fw
        adapter_text = ",".join(sorted(adapter_versions)) if adapter_versions else "-"
        print("Debugging run:", args.run_id)
        print(
            "schema_version="
            f"{schema_v}  adapter_version={adapter_text}  framework={framework or '-'}"
        )
        print(
            "Commands: list [type=.. node=.. thread=.. namespace=..] | show N | tokens N |"
            " blob N [input|output|state] | goto N | step | next [TYPE] | inject N <json> |"
            " skip N | firstdiv RUN_B | state [--pretty] | savepatch STEP FILE | lastllm |"
            " help | quit"
        )
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if line == "quit":
                break
            if line.startswith("list"):
                parts = line.split()
                filters = _parse_list_filters(parts[1:]) if len(parts) > 1 else {}
                if filters:
                    events = list(rep.iter_timeline())
                    filtered = _filter_events(
                        events,
                        etype=filters.get("type"),
                        node=filters.get("node"),
                        thread=filters.get("thread"),
                        namespace=filters.get("namespace"),
                    )
                    _print_timeline_filtered(filtered)
                else:
                    _print_timeline(rep)
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
                        # Map string to ActionType enum (e.g., "LLM")
                        at = ActionType[t]
                    except Exception:
                        print("Unknown type; use LLM|TOOL|DECISION|HITL|SNAPSHOT|SYS")
                        continue
                rep.next(at)
                print("pos=", rep._pos)
                continue
            if line.startswith("blob "):
                # blob N [input|output|state]
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
                ref = None
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
                # Move pos to first divergence vs RUN_B
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
                    "help | quit"
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
                        txt = _format_state_pretty(obj)
                        print(txt)
                    else:
                        print(obj)
                except ReplayError as e:  # pragma: no cover - interactive path
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
            if line == "snapshot":
                try:
                    ev = rep.snapshot_now()
                    print(f"Snapshot appended at step {ev.step}")
                except ReplayError as e:  # pragma: no cover - interactive path
                    print("Snapshot failed:", e)
                continue
            print("Unknown command")
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


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
            # Convert badge to string to appease static typing; Rich still renders nicely
            table.add_row(str(e.step), str(badge), e.actor, ", ".join(labels))
        console.print(table)
    except Exception:
        for e in events:
            print(f"{e.step:4d} {_plain_badge(e.action_type.value)} {e.actor:10s} {e.labels}")


def _parse_list_filters(tokens: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        k = k.strip().lower()
        v = v.strip()
        if k in {"type", "node", "thread", "namespace"} and v:
            out[k] = v
    return out


def _filter_events(
    events: list[Event],
    *,
    etype: str | None,
    node: str | None,
    thread: str | None,
    namespace: str | None,
    tool_kind: str | None = None,
    tool_name: str | None = None,
) -> list[Event]:
    def _ok(e: Event) -> bool:
        if etype and e.action_type.value != etype:
            return False
        if node and e.actor != node and e.labels.get("node") != node:
            return False
        if thread and e.labels.get("thread_id") != thread:
            return False
        if namespace and e.labels.get("namespace") != namespace:
            return False
        if tool_kind and (e.tool_kind or "") != tool_kind:
            return False
        if tool_name and (e.tool_name or "") != tool_name:
            return False
        return True

    return [e for e in events if _ok(e)]


def _format_state_pretty(
    obj: object, *, max_str: int = 200, max_items: int = 50, indent: int = 2
) -> str:
    """Pretty-format JSON-like state with truncation and size hints."""

    def _trunc(v: object) -> object:
        if isinstance(v, str):
            if len(v) > max_str:
                return v[:max_str] + f"... <truncated {len(v) - max_str} chars>"
            return v
        if isinstance(v, list):
            head = v[:max_items]
            tail = len(v) - len(head)
            lst_out = [_trunc(x) for x in head]
            if tail > 0:
                lst_out.append(f"<... {tail} more items>")
            return lst_out
        if isinstance(v, dict):
            dict_out: dict[str, object] = {}
            for k, val in v.items():
                dict_out[str(k)] = _trunc(val)
            return dict_out
        return v

    try:
        from .codec import to_bytes

        normalized = _trunc(obj)
        return to_bytes(normalized).decode("utf-8")
    except Exception:
        return repr(obj)


def dump_event_output_to_file(store: LocalStore, e: Event, path: Path) -> None:
    """Write the event's output JSON to a file.

    Prefers output_ref; falls back to input_ref if output is missing.
    Raises on IO or decode errors.
    """
    ref = e.output_ref or e.input_ref
    if not ref:
        raise ValueError("event has no output or input blob")
    raw = store.get_blob(ref)
    from .codec import from_bytes, to_bytes

    obj = from_bytes(raw)
    data = to_bytes(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _print_event(e: Event, store: LocalStore) -> None:
    print(f"Step {e.step}  Type={e.action_type.value}  Actor={e.actor}")
    # Show model/provider if available
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
    # Read the output payload and look for a chunks_ref to a BlobRef
    if not e.output_ref:
        print("No output blob for this event")
        return
    try:
        from .codec import from_bytes

        payload = from_bytes(store.get_blob(e.output_ref))
    except Exception:
        print("<failed to read output blob>")
        return
    if not isinstance(payload, dict):
        print("<output is not a JSON object>")
        return
    cref = payload.get("chunks_ref")
    if not isinstance(cref, dict):
        print("No chunks_ref found in output payload")
        return
    try:
        # Validate into BlobRef and load
        bref = BlobRef.model_validate(cref)
        raw = store.get_blob(bref)
        from .codec import from_bytes as _from_bytes

        obj = _from_bytes(raw)
        if isinstance(obj, dict) and isinstance(obj.get("chunks"), list):
            chunks = obj["chunks"]
            print(f"chunks_count={len(chunks)}")
            # Print a brief preview (first 20 chunks)
            limit = min(20, len(chunks))
            for i in range(limit):
                ch = chunks[i]
                msg = ch.get("message") if isinstance(ch, dict) else None
                if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                    print(f"{i:03d}: {msg['content']}")
                elif isinstance(msg, str):
                    print(f"{i:03d}: {msg}")
                else:
                    print(f"{i:03d}: <non-text chunk>")
            if len(chunks) > limit:
                print(f"... ({len(chunks) - limit} more)")
        else:
            print("<chunks blob missing or malformed>")
    except Exception as ex:  # pragma: no cover - defensive printing only
        print(f"<failed to load chunks_ref: {ex}>")


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
