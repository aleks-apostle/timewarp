from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from ...bindings import bind_langgraph_playback
from ...replay import LangGraphReplayer
from ...store import LocalStore
from ..helpers.imports import load_factory
from ..helpers.jsonio import dumps_text
from ..helpers.prompts import load_prompt_overrides


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
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

    from ...replay import PlaybackLLM, PlaybackMemory, PlaybackTool  # typing-only import

    # Narrow type to Callable mapping for installers signature
    prompt_overrides: dict[str, Callable[[Any], Any]] | None = None
    if getattr(args, "prompt_overrides", None):
        try:
            prompt_overrides = load_prompt_overrides(Path(str(args.prompt_overrides)))
        except Exception as exc:
            print("Failed to load prompt overrides:", exc)
            return 1

    def installer_resume(llm: PlaybackLLM, tool: PlaybackTool, memory: PlaybackMemory) -> None:
        try:
            try:
                llm.strict_meta = bool(args.strict_meta)
                llm.allow_diff = bool(getattr(args, "allow_diff", False))
                tool.strict_meta = bool(args.strict_meta)
            except Exception:
                pass
            bind_langgraph_playback(
                graph=graph,
                llm=llm,
                tool=tool,
                memory=memory,
                prompt_overrides=(None if prompt_overrides is None else dict(prompt_overrides)),
            )
        except Exception as exc:  # pragma: no cover
            print("Warning: failed to bind playback wrappers:", exc)

    replayer = LangGraphReplayer(graph=graph, store=store)
    session = replayer.resume(
        UUID(args.run_id),
        args.from_step,
        args.thread_id,
        install_wrappers=installer_resume,
        freeze_time=bool(getattr(args, "freeze_time", False)),
        no_network=bool(getattr(args, "no_network", False)),
    )
    print("Resumed run:", args.run_id)
    print("checkpoint_id=", session.checkpoint_id)
    try:
        blob_txt = dumps_text(session.result)
        print("result:")
        print(blob_txt[:2000])
    except Exception:
        print("result:", session.result)
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    res = sub.add_parser(
        "resume",
        help="Resume a run deterministically using recorded outputs",
        description=(
            "Resume a run from a prior checkpoint using playback wrappers.\n"
            "Optionally apply per-agent prompt overrides (no recording)."
        ),
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
    res.add_argument(
        "--no-network",
        dest="no_network",
        action="store_true",
        help="Block outbound network egress during replay (safety guard)",
    )
    res.add_argument(
        "--prompt-overrides",
        dest="prompt_overrides",
        default=None,
        help="Path to JSON mapping agent->override spec",
    )
    res.add_argument(
        "--allow-diff",
        dest="allow_diff",
        action="store_true",
        help="Allow prompt hash mismatches when using prompt overrides",
    )
    res.set_defaults(func=_handler)
