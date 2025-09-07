from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from ...bindings import bind_langgraph_playback
from ...replay import LangGraphReplayer
from ...store import LocalStore
from ..helpers.jsonio import dumps_text, loads_file


def _build_prompt_adapter(item: object) -> object:
    # Returns a Callable[[Any], Any] built from a JSON spec (string or object)

    def _identity(x: Any) -> Any:
        return x

    if isinstance(item, str):
        text = item

        def _adapter(x: Any) -> Any:
            if isinstance(x, list):
                return [{"role": "system", "content": text}, *list(x)]
            if isinstance(x, str):
                return str(x) + "\n\n" + text
            return x

        return _adapter
    if isinstance(item, dict):
        mode = str(item.get("mode", "prepend_system")).lower()
        text = str(item.get("text", ""))

        def _adapter(x: Any) -> Any:
            if isinstance(x, list):
                msgs = list(x)
                if mode == "append_system":
                    msgs = [*msgs, {"role": "system", "content": text}]
                elif mode == "replace_system":
                    replaced = False
                    out = []
                    for m in msgs:
                        if not replaced and isinstance(m, dict) and m.get("role") == "system":
                            out.append({"role": "system", "content": text})
                            replaced = True
                        else:
                            out.append(m)
                    msgs = out if replaced else ([{"role": "system", "content": text}, *out])
                else:
                    msgs = [{"role": "system", "content": text}, *msgs]
                return msgs
            if isinstance(x, str):
                if mode in {"append_prompt", "append_system"}:
                    return str(x) + "\n\n" + text
                elif mode == "replace_prompt":
                    return text
                else:
                    return text + "\n\n" + str(x)
            return x

        return _adapter
    return _identity


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    try:
        mod_name, func_name = args.app_factory.split(":", 1)
        mod = import_module(mod_name)
        factory = cast(Callable[[], Any], getattr(mod, func_name))
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
    prompt_overrides: dict[str, Any] | None = None
    if getattr(args, "prompt_overrides", None):
        try:
            obj = loads_file(Path(str(args.prompt_overrides)))
            if isinstance(obj, dict):
                prompt_overrides = {str(k): _build_prompt_adapter(v) for k, v in obj.items()}
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
