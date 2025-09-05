from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import import_module
from typing import Any, cast
from uuid import UUID

from ...adapters import installers as _installers
from ...replay import LangGraphReplayer
from ...store import LocalStore
from ..helpers.jsonio import dumps_text


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    try:
        mod_name, func_name = args.app_factory.split(":", 1)
        mod = import_module(mod_name)
        factory = cast(Callable[[], Any], getattr(mod, func_name))
        graph = factory()
    except Exception as exc:
        print("Failed to import app factory:", exc)
        return 1

    from ...replay import PlaybackLLM, PlaybackMemory, PlaybackTool  # typing-only import

    def installer_resume(llm: PlaybackLLM, tool: PlaybackTool, memory: PlaybackMemory) -> None:
        try:
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
    res.set_defaults(func=_handler)
