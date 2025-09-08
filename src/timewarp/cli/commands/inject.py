from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from ...bindings import bind_langgraph_playback
from ...events import Run as _Run
from ...replay import LangGraphReplayer
from ...store import LocalStore
from ...utils.logging import log_warn_once
from ..helpers.imports import load_factory
from ..helpers.jsonio import loads_file
from ..helpers.prompts import load_prompt_overrides


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    # Validate mutually exclusive modes
    modes = [bool(args.output_file), bool(args.state_patch_file), bool(args.prompt_overrides)]
    if sum(1 for m in modes if m) != 1:
        print("Provide exactly one of --output, --state-patch, or --prompt-overrides")
        return 1

    replacement: object | None = None
    patch_obj: object | None = None
    if args.output_file:
        try:
            replacement = loads_file(Path(args.output_file))
        except Exception as exc:
            print("Failed to read replacement output:", exc)
            return 1
    if args.state_patch_file:
        try:
            patch_obj = loads_file(Path(args.state_patch_file))
        except Exception as exc:
            print("Failed to read state patch:", exc)
            return 1

    prompt_overrides: dict[str, Callable[[Any], Any]] | None = None
    if args.prompt_overrides:
        try:
            prompt_overrides = load_prompt_overrides(Path(args.prompt_overrides))
        except Exception as exc:
            print("Failed to read prompt overrides:", exc)
            return 1

    try:
        factory = load_factory(args.app_factory)
        graph = factory()
    except Exception as exc:
        print("Failed to import app factory:", exc)
        return 1

    def _assert_langgraph(obj: object) -> None:
        if not (hasattr(obj, "stream") or hasattr(obj, "invoke")):
            raise SystemExit(
                "This CLI only supports LangGraph compiled graphs (need .stream/.invoke)"
            )

    _assert_langgraph(graph)

    from ...replay import PlaybackLLM, PlaybackMemory, PlaybackTool  # typing-only imports

    teardowns: list[Callable[[], None]] = []

    def installer_inject(llm: PlaybackLLM, tool: PlaybackTool, memory: PlaybackMemory) -> None:
        try:
            try:
                llm.strict_meta = bool(args.strict_meta)
                llm.allow_diff = bool(getattr(args, "allow_diff", False))
                tool.strict_meta = bool(args.strict_meta)
            except Exception:
                pass
            td = bind_langgraph_playback(
                graph=graph,
                llm=llm,
                tool=tool,
                memory=memory,
                prompt_overrides=(None if prompt_overrides is None else dict(prompt_overrides)),
            )
            teardowns.append(td)
        except Exception as exc:  # pragma: no cover
            print("Warning: failed to bind playback wrappers:", exc)

    # State patch mode
    if patch_obj is not None:
        try:
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

    # Output override mode or prompt overrides fork
    replayer = LangGraphReplayer(graph=graph, store=store)
    if prompt_overrides is not None:
        new_id = replayer.fork_with_prompt_overrides(
            UUID(args.run_id),
            prompt_overrides,
            args.thread_id,
            install_wrappers=installer_inject,
            freeze_time=bool(getattr(args, "freeze_time", False)),
            allow_diff=bool(getattr(args, "allow_diff", False)),
        )
    else:
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
            # For now, fork recording is supported for LangGraph only
            adapter_name = str(getattr(args, "adapter", "auto") or "auto").lower()
            if adapter_name not in ("auto", "langgraph"):
                print("--record-fork currently supports only the 'langgraph' adapter")
                return 1
            # Retrieve original input payload
            evs = store.list_events(UUID(args.run_id))
            orig_input = None
            for ev in evs:
                if ev.input_ref is not None:
                    from ...codec import from_bytes as _from_bytes

                    orig_input = _from_bytes(store.get_blob(ev.input_ref))
                    break
            if orig_input is None:
                print("Could not locate original input for the run; aborting fork recording")
                return 1
            # Compose new Run metadata: branch_of + override_step
            # Try to copy project/name from original run metadata when available
            proj = None
            name = None
            for r in store.list_runs():
                if r.run_id == UUID(args.run_id):
                    proj = r.project
                    name = r.name
                    break
            new_labels = {"branch_of": str(args.run_id)}
            if prompt_overrides is not None:
                new_labels["override_step"] = "prompt_overrides"
            else:
                new_labels["override_step"] = str(args.step)
            new_run = _Run(
                run_id=new_id,
                project=proj,
                name=name,
                framework="langgraph",
                labels=new_labels,
            )
            from ...langgraph import LangGraphRecorder as _LGRecorder

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
                except Exception as e:
                    log_warn_once("cli.inject.teardown_failed", e)
    else:
        if prompt_overrides is not None:
            print("Fork prepared with prompt overrides")
        else:
            print("Fork prepared with override at step", args.step)
        print("New run id:", new_id)
        print(
            "Note: to record the fork immediately, re-run with --record-fork or "
            "run your app with the recorder."
        )
        return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    inj = sub.add_parser("inject")
    inj.add_argument("run_id", help="Run ID")
    inj.add_argument(
        "step", type=int, help="Step to override (ignored with --state-patch/--prompt-overrides)"
    )
    inj.add_argument("--output", dest="output_file", help="JSON file with replacement output")
    inj.add_argument(
        "--state-patch",
        dest="state_patch_file",
        help="JSON file with state patch to apply at latest checkpoint",
    )
    inj.add_argument(
        "--prompt-overrides",
        dest="prompt_overrides",
        default=None,
        help="Path to JSON mapping agent->override spec",
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
        "--allow-diff",
        dest="allow_diff",
        action="store_true",
        help="Allow prompt hash mismatches when using prompt overrides",
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
    inj.set_defaults(func=_handler)
