#!/usr/bin/env python3
"""
Interactive debugger for Timewarp + LangGraph (packaged).

- CLI: `timewarp-repl` (via project.scripts)
- Library: programmatic REPL via `launch_debugger(...)`.
- Accepts a compiled LangGraph object or a factory string.
- Unifies timeline/event inspection, prompt/tools/memory views,
  deterministic resume, what-if injection and fork recording, and diffs.
"""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from timewarp.codec import from_bytes
from timewarp.diff import bisect_divergence, first_divergence
from timewarp.events import ActionType, Event, Run
from timewarp.memory import rebuild_memory_snapshot
from timewarp.replay import LangGraphReplayer, PlaybackLLM, PlaybackMemory, PlaybackTool
from timewarp.store import LocalStore

# Optional helper utilities (semi-internal; stable in this repo)
try:
    from timewarp.cli.helpers.imports import load_factory  # to resolve module:function
    from timewarp.cli.helpers.prompts import build_prompt_adapter  # compile prompt overrides
except Exception:  # pragma: no cover - optional extras
    load_factory = None  # type: ignore[assignment]
    build_prompt_adapter = None  # type: ignore[assignment]

# Optional pretty UI (rich) — alias names to avoid pyright possiblyUnboundVariable
if TYPE_CHECKING:  # for type checkers
    from rich.console import Console as RichConsole
    from rich.table import Table as RichTableType
    from rich.text import Text as RichTextType
else:  # at runtime, attempt import; fall back to simple printing
    RichConsole = Any  # type: ignore[misc, assignment]
    RichTableType = Any  # type: ignore[misc, assignment]
    RichTextType = Any  # type: ignore[misc, assignment]

# Explicit annotations to satisfy static checkers
RichTable: type[Any] | None = None
RichText: type[Any] | None = None
console: Any | None = None

try:  # best-effort rich import
    from rich.console import Console as _Console
    from rich.table import Table as _RichTable
    from rich.text import Text as _RichText

    RICH = True
    console = _Console()
    RichTable = _RichTable
    RichText = _RichText
except Exception:  # pragma: no cover - optional dependency
    RICH = False
    console = None
    RichTable = None
    RichText = None


@dataclass
class DebugConfig:
    db: Path
    blobs: Path
    run_id: UUID
    thread_id: str | None = None
    freeze_time: bool = False
    strict_meta: bool = False
    allow_diff: bool = False
    prompt_overrides: dict[str, Callable[[Any], Any]] | None = None


@dataclass
class Debugger:
    store: LocalStore
    run_id: UUID
    graph: Any | None = None  # compiled LangGraph
    cfg: DebugConfig = field(
        default_factory=lambda: DebugConfig(
            Path("./timewarp.sqlite3"), Path("./blobs"), UUID(int=0)
        )
    )
    _last_pos: int = 0  # basic cursor for 'llm' convenience

    # --------- High-level actions ---------

    def print_header(self) -> None:
        evs = self.store.list_events(self.run_id)
        if not evs:
            print("No events for this run_id")
            return
        schema_v = evs[0].schema_version
        adapter_versions = set()
        framework = None
        for e in evs:
            mm = e.model_meta or {}
            if isinstance(mm, dict):
                av = mm.get("adapter_version")
                if isinstance(av, str):
                    adapter_versions.add(av)
                if framework is None and isinstance(mm.get("framework"), str):
                    framework = mm.get("framework")
        print(f"Run: {self.run_id}")
        adapters = ", ".join(sorted(adapter_versions)) or "-"
        fw = framework or "-"
        print(f"Schema: v{schema_v}  Adapter(s): {adapters}  Framework: {fw}")
        print(f"Events: {len(evs)}  DB: {self.store.db_path}  Blobs: {self.store.blobs_root}")

    def list_timeline(
        self, *, filters: dict[str, str] | None = None, page: int | None = None, page_size: int = 50
    ) -> None:
        events = self.store.list_events(self.run_id)
        if filters:
            events = self._filter_events(events, **filters)
        if page is not None:
            start = max(0, page * page_size)
            end = min(len(events), start + page_size)
            events = events[start:end]
        if RICH and RichTable is not None:
            table = RichTable(title=f"Timeline ({len(events)})")
            table.add_column("Step", justify="right")
            table.add_column("Type")
            table.add_column("Actor")
            table.add_column("Labels")
            for e in events:
                table.add_row(
                    str(e.step),
                    self._badge(e.action_type.value),
                    str(e.actor),
                    self._format_labels(e),
                )
            if console is not None:
                console.print(table)
        else:
            for e in events:
                labels = self._format_labels(e)
                print(f"{e.step:>5}  {e.action_type.value:<9}  {e.actor:<20}  {labels}")

    def show_event(self, step: int) -> None:
        ev = self._by_step(step)
        if not ev:
            print("No such step")
            return
        self._print_event(ev)
        self._last_pos = int(step)

    def show_last_llm(self) -> None:
        events = self.store.list_events(self.run_id)
        # If no cursor set yet, show the last LLM in the run
        if self._last_pos <= 0:
            for e in reversed(events):
                if e.action_type is ActionType.LLM:
                    self._print_event(e)
                    return
            print("No LLM event found")
            return
        # Otherwise, find the last LLM at/before cursor
        idx = len(events) - 1
        while idx >= 0:
            e = events[idx]
            if e.step <= self._last_pos and e.action_type is ActionType.LLM:
                self._print_event(e)
                return
            idx -= 1
        print("No LLM event found at/before current position")

    def prompt(self, step: int | None = None) -> None:
        e = self._pick_llm_step(step)
        if e is None:
            print("No LLM event at that step")
            return
        # Prompt/message context is stored on the LLM event's input_ref when available
        payload = self._read_json_blob(e.input_ref) if e.input_ref else None
        if not isinstance(payload, dict):
            print("<output is not a JSON object>")
            return
        messages = payload.get("messages")
        tools = payload.get("tools")
        # print messages (head)
        try:
            if isinstance(messages, list):
                head = messages[:10]
                print(f"messages_count={len(messages)}")
                for i, m in enumerate(head):
                    print(f"  - messages[{i}]: {m}")
                if len(messages) > len(head):
                    print(f"  ... ({len(messages) - len(head)} more)")
        except Exception:
            pass
        # print tools (head)
        try:
            if isinstance(tools, list):
                t_head = tools[:10]
                print(f"tools_count={len(tools)}")
                for i, t in enumerate(t_head):
                    print(f"  - tool[{i}]: {t}")
                if len(tools) > len(t_head):
                    print(f"  ... ({len(tools) - len(t_head)} more)")
        except Exception:
            pass
        est_tokens = self._estimate_tokens(messages, tools)
        print(f"estimated_tokens≈{est_tokens}")

    def tools(self, step: int | None = None) -> None:
        events = self.store.list_events(self.run_id)
        if step is None:
            # Summary across all LLM steps
            rows: list[tuple[int, str, str, int, str]] = []
            for e in events:
                if e.action_type is not ActionType.LLM:
                    continue
                tools, digest = self._extract_tools_from_llm_event(e)
                rows.append(
                    (e.step, e.actor, e.labels.get("thread_id", ""), len(tools or []), digest or "")
                )
            if RICH and RichTable is not None and console is not None:
                table = RichTable(title="Tools Summary")
                table.add_column("Step", justify="right")
                table.add_column("Actor")
                table.add_column("Thread")
                table.add_column("#Tools", justify="right")
                table.add_column("Digest")
                for step_i, actor, thr, count, dg in rows:
                    table.add_row(str(step_i), actor, thr, str(count), dg)
                console.print(table)
            else:
                for step_i, actor, thr, count, dg in rows:
                    line = (
                        f"{step_i:>5}  actor={actor:<16}  thread={thr:<10}  "
                        f"tools={count:<3}  digest={dg}"
                    )
                    print(line)
            return
        llm_ev = self._pick_llm_step(step)
        if llm_ev is None:
            print("No LLM event at that step")
            return
        tools, digest = self._extract_tools_from_llm_event(llm_ev)
        print(
            f"LLM step: {llm_ev.step}  actor={llm_ev.actor}  "
            f"thread={llm_ev.labels.get('thread_id', '')}"
        )
        print(f"tools_digest={digest or ''}")
        print(f"available_tools={len(tools or [])}")
        if tools:
            for i, t in enumerate(tools[:10]):
                print(f"  - tool[{i}]: {t}")
            if len(tools) > 10:
                print(f"  ... ({len(tools) - 10} more)")

    def memory_list(self) -> None:
        # Print a compact summary of memory providers and scopes observed so far
        snap = rebuild_memory_snapshot(self.store, self.run_id, step=10**9)  # up to end
        by_space = snap.get("by_space", {})
        if RICH and RichTable is not None and console is not None:
            table = RichTable(title="Memory Summary")
            table.add_column("Space")
            table.add_column("Scopes")
            for space, scopes in by_space.items():
                table.add_row(space, ", ".join(sorted(scopes.keys())))
            console.print(table)
        else:
            for space, scopes in by_space.items():
                print(f"{space}: {', '.join(sorted(scopes.keys()))}")

    def memory_show(self, step: int) -> None:
        snap = rebuild_memory_snapshot(
            self.store, self.run_id, step=step, thread_id=self.cfg.thread_id
        )
        print(json.dumps(snap, ensure_ascii=False, indent=2))

    def memory_diff(self, a_step: int, b_step: int, key_path: str | None) -> None:
        # Diff two memory snapshots; optionally restrict to a dotted key path
        from timewarp.utils.diffing import struct_diff  # local import

        a = rebuild_memory_snapshot(
            self.store, self.run_id, step=a_step, thread_id=self.cfg.thread_id
        )
        b = rebuild_memory_snapshot(
            self.store, self.run_id, step=b_step, thread_id=self.cfg.thread_id
        )
        if key_path:
            a = self._pluck_path(a, key_path)
            b = self._pluck_path(b, key_path)
        diff = struct_diff(a, b)
        print(json.dumps(diff, ensure_ascii=False, indent=2))

    def resume(self, from_step: int | None = None) -> None:
        if self.graph is None:
            print(
                "No graph bound. Use `app module:function` or pass `graph=` to launch_debugger()."
            )
            return

        def installer(llm: PlaybackLLM, tool: PlaybackTool, memory: PlaybackMemory) -> None:
            try:
                llm.strict_meta = bool(self.cfg.strict_meta)
                llm.allow_diff = bool(self.cfg.allow_diff)
                tool.strict_meta = bool(self.cfg.strict_meta)
            except Exception:
                pass
            try:
                from timewarp.bindings import bind_langgraph_playback

                bind_langgraph_playback(
                    graph=self.graph,
                    llm=llm,
                    tool=tool,
                    memory=memory,
                    prompt_overrides=(
                        None
                        if self.cfg.prompt_overrides is None
                        else dict(self.cfg.prompt_overrides)
                    ),
                )
            except Exception as exc:
                print("Warning: failed to bind playback wrappers:", exc)

        replayer = LangGraphReplayer(graph=self.graph, store=self.store)
        session = replayer.resume(
            run_id=self.run_id,
            from_step=from_step,
            thread_id=self.cfg.thread_id,
            install_wrappers=installer,
            freeze_time=self.cfg.freeze_time,
        )
        print("Replayed result:", repr(session.result))

    def inject(
        self,
        step: int,
        output: Any | None,
        *,
        record_fork: bool = False,
        overrides_file: Path | None = None,
    ) -> None:
        if self.graph is None:
            print(
                "No graph bound. Use `app module:function` or pass `graph=` to launch_debugger()."
            )
            return
        # Validate mutually exclusive modes
        modes = [output is not None, overrides_file is not None]
        if sum(1 for m in modes if m) != 1:
            print("Provide exactly one of: output JSON OR prompt-overrides file")
            return

        # Build wrappers with either one-shot output override or prompt overrides
        teardowns: list[Callable[[], None]] = []

        def installer(llm: PlaybackLLM, tool: PlaybackTool, memory: PlaybackMemory) -> None:
            try:
                llm.strict_meta = bool(self.cfg.strict_meta)
                llm.allow_diff = bool(self.cfg.allow_diff)
                tool.strict_meta = bool(self.cfg.strict_meta)
            except Exception:
                pass
            try:
                from timewarp.bindings import bind_langgraph_playback

                po = dict(self.cfg.prompt_overrides or {})
                if overrides_file is not None:
                    po = self._load_prompt_overrides_file(overrides_file)
                td = bind_langgraph_playback(
                    graph=self.graph,
                    llm=llm,
                    tool=tool,
                    memory=memory,
                    prompt_overrides=po if po else None,
                )
                teardowns.append(td)
            except Exception as exc:
                print("Warning: failed to bind playback wrappers:", exc)

        replayer = LangGraphReplayer(graph=self.graph, store=self.store)
        if overrides_file is not None:
            new_id = replayer.fork_with_prompt_overrides(
                self.run_id,
                self._load_prompt_overrides_file(overrides_file),
                self.cfg.thread_id,
                install_wrappers=installer,
                freeze_time=self.cfg.freeze_time,
                allow_diff=self.cfg.allow_diff,
            )
        else:
            new_id = replayer.fork_with_injection(
                self.run_id,
                at_step=step,
                replacement=output,
                thread_id=self.cfg.thread_id,
                install_wrappers=installer,
                freeze_time=self.cfg.freeze_time,
            )

        if not record_fork:
            print("Fork prepared. New run id:", new_id)
            print("Tip: run `resume` with your app recorder to execute & persist the branch.")
            return

        # Record immediately by invoking a new Run under a recorder.
        # Retrieve original input payload (first available input_ref)
        evs = self.store.list_events(self.run_id)
        orig_input = None
        for e in evs:
            if e.input_ref is not None:
                orig_input = from_bytes(self.store.get_blob(e.input_ref))
                break
        if orig_input is None:
            print("Could not locate original input for the run; aborting fork recording")
            return

        # Copy metadata for lineage
        proj = None
        name = None
        for r in self.store.list_runs():
            if r.run_id == self.run_id:
                proj = r.project
                name = r.name
                break
        new_labels = {"branch_of": str(self.run_id)}
        new_labels["override_step"] = (
            "prompt_overrides" if overrides_file is not None else str(step)
        )

        new_run = Run(
            run_id=new_id, project=proj, name=name, framework="langgraph", labels=new_labels
        )
        self.store.create_run(new_run)

        from timewarp.langgraph import LangGraphRecorder

        rec = LangGraphRecorder(
            graph=self.graph,
            store=self.store,
            run=new_run,
            snapshot_every=20,
            stream_modes=("updates", "messages", "values"),
            stream_subgraphs=True,
        )
        cfg2: dict[str, Any] = {"configurable": {}}
        if self.cfg.thread_id is not None:
            cfg2 = {"configurable": {"thread_id": self.cfg.thread_id}}
        try:
            _ = rec.invoke(orig_input, config=cfg2)
        finally:
            # best-effort teardown of any monkeypatches
            for td in teardowns:
                try:
                    td()
                except Exception:
                    pass
        print("Fork executed and recorded:", new_id)

    def diff(self, other_run: UUID, *, window: int = 5, bisect: bool = False) -> None:
        if bisect:
            res = bisect_divergence(self.store, self.run_id, other_run, window=window)
            payload: dict[str, int | str] | dict[str, str]
            payload = res if res is not None else {"result": ""}
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return
        div = first_divergence(self.store, self.run_id, other_run, window=window)
        if not div:
            print("No divergence detected")
            return
        print(f"First divergence: A@{div.step_a} vs B@{div.step_b}  reason={div.reason}")

    # --------- REPL ---------

    def repl(self) -> None:
        self.print_header()
        print("Type 'help' for a list of commands. Ctrl-D or 'exit' to quit.")
        while True:
            try:
                line = input("tw> ").strip()
            except EOFError:
                print()
                break
            if not line:
                continue
            try:
                cmd, *args = shlex.split(line)
            except Exception:
                print("Parse error")
                continue
            if cmd in ("exit", "quit", "q"):
                break
            if cmd in ("help", "?"):
                self._print_help()
                continue
            # Mutators / setup
            if cmd == "app":
                if not args:
                    print("Usage: app module:function")
                    continue
                spec = args[0]
                if load_factory is None:
                    print("load_factory unavailable; install timewarp extras or run as a library.")
                    continue
                try:
                    factory = load_factory(spec)
                    self.graph = factory()
                    print("Graph loaded.")
                except Exception as exc:
                    print("Failed to load app:", exc)
                continue
            if cmd == "thread":
                self.cfg.thread_id = args[0] if args else None
                print("Thread set:", self.cfg.thread_id)
                continue
            if cmd == "freeze":
                self.cfg.freeze_time = True
                print("Freeze-time ON")
                continue
            if cmd == "unfreeze":
                self.cfg.freeze_time = False
                print("Freeze-time OFF")
                continue
            if cmd == "strict":
                self.cfg.strict_meta = True
                print("Strict meta ON")
                continue
            if cmd == "nonstrict":
                self.cfg.strict_meta = False
                print("Strict meta OFF")
                continue
            if cmd == "allowdiff":
                self.cfg.allow_diff = True
                print("Allow prompt diffs ON")
                continue
            if cmd == "disallowdiff":
                self.cfg.allow_diff = False
                print("Allow prompt diffs OFF")
                continue
            if cmd == "overrides":
                if not args:
                    self.cfg.prompt_overrides = None
                    print("Cleared prompt overrides.")
                else:
                    p = Path(args[0])
                    self.cfg.prompt_overrides = self._load_prompt_overrides_file(p)
                    count = len(self.cfg.prompt_overrides or {})
                    print(f"Loaded prompt overrides for {count} agent(s).")
                continue
            # Views
            if cmd == "list":
                filters = self._parse_filters(args)
                self.list_timeline(filters=filters)
                continue
            if cmd == "event":
                if not args:
                    print("Usage: event STEP")
                    continue
                try:
                    step = int(args[0])
                except Exception:
                    print("STEP must be an integer")
                    continue
                self.show_event(step)
                continue
            if cmd == "llm":
                self.show_last_llm()
                continue
            if cmd == "prompt":
                step2 = None
                if args:
                    try:
                        step2 = int(args[0])
                    except Exception:
                        print("Usage: prompt [STEP]")
                        continue
                self.prompt(step2)
                if step2 is not None:
                    self._last_pos = int(step2)
                continue
            if cmd == "tools":
                step3 = None
                if args:
                    try:
                        step3 = int(args[0])
                    except Exception:
                        print("Usage: tools [STEP]")
                        continue
                self.tools(step3)
                if step3 is not None:
                    self._last_pos = int(step3)
                continue
            if cmd == "memory":
                self.memory_list()
                continue
            if cmd == "memory_show":
                if not args:
                    print("Usage: memory_show STEP")
                    continue
                self.memory_show(int(args[0]))
                continue
            if cmd == "memory_diff":
                if len(args) < 2:
                    print("Usage: memory_diff A_STEP B_STEP [key=dot.path]")
                    continue
                a = int(args[0])
                b = int(args[1])
                key = None
                if len(args) >= 3 and args[2].startswith("key="):
                    key = args[2].split("=", 1)[1]
                self.memory_diff(a, b, key)
                continue
            # Execution
            if cmd == "resume":
                from_step = None
                if args:
                    try:
                        from_step = int(args[0])
                    except Exception:
                        print("Usage: resume [FROM_STEP]")
                        continue
                self.resume(from_step)
                continue
            if cmd == "inject":
                if not args:
                    print("Usage: inject STEP output.json [--record]")
                    continue
                step4 = int(args[0])
                out_file = Path(args[1])
                try:
                    output = json.loads(out_file.read_text(encoding="utf-8"))
                except Exception as exc:
                    print("Failed to read output JSON:", exc)
                    continue
                record = "--record" in args or "-r" in args
                self.inject(step4, output, record_fork=record)
                continue
            if cmd == "fork_prompts":
                if not args:
                    print("Usage: fork_prompts overrides.json [--record]")
                    continue
                file = Path(args[0])
                record2 = "--record" in args or "-r" in args
                self.inject(step=0, output=None, record_fork=record2, overrides_file=file)
                continue
            if cmd == "diff":
                if not args:
                    print("Usage: diff OTHER_RUN_ID [--bisect] [--window N]")
                    continue
                other = UUID(args[0])
                do_bisect = "--bisect" in args
                win = 5
                for tok in args[1:]:
                    if tok.startswith("--window"):
                        try:
                            win = int(tok.split("=")[1])
                        except Exception:
                            pass
                self.diff(other, window=win, bisect=do_bisect)
                continue

            print("Unknown command. Type 'help'.")

    # --------- small helpers ---------

    def _print_help(self) -> None:
        print(
            """
Commands
  app module:function         Load a compiled LangGraph via factory (enables resume/inject)
  thread T                    Set thread_id to use during resume/inject
  freeze | unfreeze           Toggle freeze-time replay
  strict | nonstrict          Toggle strict meta checks (provider/model/tools invariants)
  allowdiff | disallowdiff    Toggle allowing prompt diffs during replay (for overrides)
  overrides [file.json]       Load per-agent prompt overrides; empty to clear

Views
  list [type=LLM|TOOL|...] [node=...] [thread=...] [ns=...]     Timeline (filterable)
  event STEP                  Show a single event (and blob size hints)
  llm                         Show the last LLM event before current position
  prompt [STEP]               Messages/tools head + estimated tokens for an LLM step
  tools [STEP]                Tools summary across run or details for one LLM step
  memory                      Memory summary by space
  memory_show STEP            Full memory snapshot up to step
  memory_diff A B [key=path]  Structural diff between two snapshots (optional dotted key)

Execution
  resume [FROM_STEP]          Deterministically resume from a checkpoint using recorded outputs
  inject STEP output.json [-r|--record]
                              One-shot override at STEP; optionally record fork immediately
  fork_prompts overrides.json [-r|--record]
                              Prepare/record a fork that applies prompt overrides
  diff OTHER_RUN_ID [--bisect] [--window N]
                              Show first divergence or minimal failing window
"""
        )

    def _badge(self, kind: str) -> str | Any:
        if not RICH or RichText is None:
            return kind
        try:
            styles = {
                "LLM": "bold cyan",
                "TOOL": "bold magenta",
                "HITL": "bold yellow",
                "SNAPSHOT": "bold green",
                "SYS": "dim",
                "DECISION": "bold blue",
                "ERROR": "bold red",
                "MEMORY": "bold white",
                "RETRIEVAL": "bold white",
            }
            return RichText(kind, style=styles.get(kind, ""))
        except Exception:
            return kind

    def _format_labels(self, e: Event) -> str:
        try:
            labels = []
            mm = e.model_meta or {}
            sm = mm.get("adapter_version") if isinstance(mm, dict) else None
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

    def _print_event(self, e: Event) -> None:
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
        if e.input_ref:
            print("input_blob:", e.input_ref.path)
        if e.output_ref:
            print("output_blob:", e.output_ref.path)
            try:
                obj = self._read_json_blob(e.output_ref)
                size = len(json.dumps(obj, ensure_ascii=False))
                print(f"output_json_size≈{size} bytes")
            except Exception:
                pass

    def _read_json_blob(self, ref: Any) -> Any:
        raw = self.store.get_blob(ref)
        return from_bytes(raw)

    def _pick_llm_step(self, step: int | None) -> Event | None:
        events = self.store.list_events(self.run_id)
        e: Event | None = None
        if step is None:
            for ev in reversed(events):
                if ev.action_type is ActionType.LLM:
                    e = ev
                    break
        else:
            e = next(
                (x for x in events if x.step == step and x.action_type is ActionType.LLM), None
            )
        return e

    def _by_step(self, step: int) -> Event | None:
        try:
            s = int(step)
        except Exception:
            return None
        for ev in self.store.list_events(self.run_id):
            if int(ev.step) == s:
                return ev
        return None

    def _estimate_tokens(self, messages: Any, tools: Any) -> int:
        # provider-agnostic heuristic: characters/4 plus a tool-list penalty
        try:
            chars = 0
            if isinstance(messages, list):
                for m in messages:
                    if isinstance(m, dict):
                        for _k, v in m.items():
                            if isinstance(v, str):
                                chars += len(v)
            elif isinstance(messages, str):
                chars += len(messages)
            t_penalty = 0
            if isinstance(tools, list):
                t_penalty = sum(len(json.dumps(t)) for t in tools[:20]) // 4
            return (chars // 4) + t_penalty
        except Exception:
            return 0

    def _extract_tools_from_llm_event(self, e: Event) -> tuple[list[object] | None, str | None]:
        # Prefer hashes["tools"], then explicit tools_digest field
        tools_digest: str | None = None
        try:
            if e.hashes and isinstance(e.hashes.get("tools"), str):
                tools_digest = e.hashes["tools"]
        except Exception:
            tools_digest = None
        if tools_digest is None:
            try:
                tools_digest = e.tools_digest
            except Exception:
                tools_digest = None
        # Tools list lives in the input_ref context blob when present
        try:
            payload = self._read_json_blob(e.input_ref) if e.input_ref else None
        except Exception:
            payload = None
        tools = None
        if isinstance(payload, dict):
            maybe_tools = payload.get("tools")
            if isinstance(maybe_tools, list):
                tools = maybe_tools
        return tools, tools_digest

    def _pluck_path(self, obj: Any, dotted: str | None) -> Any:
        if not dotted:
            return obj
        cur = obj
        try:
            for seg in str(dotted).split("."):
                if isinstance(cur, dict) and seg in cur:
                    cur = cur[seg]
                else:
                    return cur
        except Exception:
            return cur
        return cur

    def _filter_events(
        self,
        events: Iterable[Event],
        *,
        etype: str | None = None,
        node: str | None = None,
        thread: str | None = None,
        ns: str | None = None,
    ) -> list[Event]:
        out: list[Event] = []
        for e in events:
            if etype and e.action_type.value != etype:
                continue
            if node and e.actor != node and e.labels.get("node") != node:
                continue
            if thread and e.labels.get("thread_id") != thread:
                continue
            if ns and e.labels.get("namespace") != ns:
                continue
            out.append(e)
        return out

    def _parse_filters(self, args: list[str]) -> dict[str, str]:
        filt: dict[str, str] = {}
        for a in args:
            if "=" in a:
                k, v = a.split("=", 1)
                if k in ("type", "node", "thread", "ns", "namespace"):
                    if k == "namespace":
                        k = "ns"
                    filt[k] = v
        return filt

    def _load_prompt_overrides_file(self, path: Path) -> dict[str, Callable[[Any], Any]]:
        if build_prompt_adapter is None:
            raise RuntimeError("Prompt override helpers unavailable; install library extras.")
        spec = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(spec, dict):
            raise ValueError("overrides file must be an object mapping agent->spec")
        out: dict[str, Callable[[Any], Any]] = {}
        for agent, item in spec.items():
            out[str(agent)] = build_prompt_adapter(item)
        return out


# ---- programmatic entrypoint ----


def launch_debugger(
    *,
    db: str | Path,
    blobs: str | Path,
    run_id: str | UUID,
    graph: Any | None = None,
    thread_id: str | None = None,
    freeze_time: bool = False,
    strict_meta: bool = False,
    allow_diff: bool = False,
    prompt_overrides: dict[str, Callable[[Any], Any]] | None = None,
) -> None:
    """Launch the interactive debugger programmatically."""
    run_id_uuid = UUID(str(run_id))
    store = LocalStore(db_path=Path(db), blobs_root=Path(blobs))
    cfg = DebugConfig(
        db=Path(db),
        blobs=Path(blobs),
        run_id=run_id_uuid,
        thread_id=thread_id,
        freeze_time=freeze_time,
        strict_meta=strict_meta,
        allow_diff=allow_diff,
        prompt_overrides=prompt_overrides,
    )
    dbg = Debugger(store=store, run_id=run_id_uuid, graph=graph, cfg=cfg)
    dbg.repl()


# ---- CLI entrypoint ----


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Interactive debugger for Timewarp + LangGraph")
    p.add_argument("db", nargs="?", default="./timewarp.sqlite3", help="SQLite DB path")
    p.add_argument("blobs", nargs="?", default="./blobs", help="Blobs root")
    p.add_argument("run_id", help="Run UUID")
    p.add_argument(
        "--app", dest="app_factory", help="module:function that returns a compiled graph"
    )
    p.add_argument("--thread", dest="thread_id", help="LangGraph thread_id")
    p.add_argument("--freeze-time", action="store_true", help="Freeze time to recorded timestamps")
    p.add_argument(
        "--strict-meta", action="store_true", help="Enforce provider/model/tools invariants"
    )
    p.add_argument(
        "--allow-diff", action="store_true", help="Allow prompt diffs during replay (for overrides)"
    )
    p.add_argument(
        "--overrides", dest="overrides_file", help="JSON file with per-agent prompt overrides"
    )
    args = p.parse_args(argv)

    store = LocalStore(db_path=Path(args.db), blobs_root=Path(args.blobs))
    graph = None
    po = None
    if args.app_factory:
        if load_factory is None:
            print("--app requires timewarp.cli helpers; install full package.")
            return 2
        try:
            factory = load_factory(args.app_factory)
            graph = factory()
        except Exception as exc:
            print("Failed to import app factory:", exc)
            return 2
    if args.overrides_file:
        if build_prompt_adapter is None:
            print("--overrides requires prompt helpers; install full package.")
            return 2
        try:
            spec = json.loads(Path(args.overrides_file).read_text(encoding="utf-8"))
            if not isinstance(spec, dict):
                raise ValueError("overrides file must be a JSON object mapping agent->spec")
            po = {str(k): build_prompt_adapter(v) for k, v in spec.items()}
        except Exception as exc:
            print("Failed to read overrides file:", exc)
            return 2

    cfg = DebugConfig(
        db=Path(args.db),
        blobs=Path(args.blobs),
        run_id=UUID(args.run_id),
        thread_id=args.thread_id,
        freeze_time=bool(args.freeze_time),
        strict_meta=bool(args.strict_meta),
        allow_diff=bool(args.allow_diff),
        prompt_overrides=po,
    )
    dbg = Debugger(store=store, run_id=cfg.run_id, graph=graph, cfg=cfg)
    dbg.repl()
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())
