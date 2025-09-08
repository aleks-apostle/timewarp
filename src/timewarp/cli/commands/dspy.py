from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from uuid import UUID

from ...exporters.dspy import build_dspy_dataset
from ...store import LocalStore
from ..helpers.imports import load_factory
from ..helpers.jsonio import dumps_text, loads_file, print_json
from ..helpers.prompts import load_prompt_overrides


def _handler_build(args: argparse.Namespace, store: LocalStore) -> int:
    run_id = UUID(args.run_id)
    agents: list[str] | None
    if getattr(args, "agents", None):
        agents = [s for s in str(args.agents).split(",") if s]
    else:
        agents = None
    ds = build_dspy_dataset(store, run_id, agents=agents)
    out = getattr(args, "out", None)
    if out:
        Path(out).write_text(dumps_text(ds), encoding="utf-8")
        print(f"Wrote dataset to {out}")
        return 0
    print_json(ds)
    return 0


def _heuristic_prompt_for_agent(agent: str, examples: list[dict[str, Any]], k: int = 3) -> str:
    head = examples[: max(1, min(k, len(examples)))]
    lines: list[str] = []
    lines.append(f"You are the '{agent}' agent. Use memory and messages to produce an answer.")
    lines.append("Follow the style of the outputs in the examples.")
    for i, ex in enumerate(head):
        try:
            msgs = ex.get("inputs", {}).get("messages")
            mem = ex.get("memory", {})
            out = ex.get("output")
            mem_keys = ", ".join(sorted(list(mem.keys()))) if isinstance(mem, dict) else ""
            lines.append(f"\nExample {i + 1}:")
            if isinstance(msgs, list):
                snippet = []
                for m in msgs[-3:]:
                    try:
                        role = m.get("role") if isinstance(m, dict) else None
                        content = m.get("content") if isinstance(m, dict) else None
                        snippet.append(f"{role}: {content}")
                    except Exception:
                        continue
                if snippet:
                    lines.append("Messages:\n" + "\n".join(snippet))
            if mem_keys:
                lines.append(f"Memory keys: {mem_keys}")
            if isinstance(out, str | int | float):
                lines.append(f"Output: {out}")
        except Exception:
            continue
    lines.append("\nWhen you answer, be concise and accurate.")
    return "\n".join(lines)


def _handler_optimize(args: argparse.Namespace, _store: LocalStore) -> int:
    ds_path = Path(args.dataset)
    data = loads_file(ds_path)
    if not isinstance(data, dict):
        print("Invalid dataset JSON: expected object mapping agent -> examples")
        return 1

    # Optional DSPy integration; gracefully fall back to heuristics when missing.
    use_optimizer = str(getattr(args, "optimizer", "none")).lower()
    results: dict[str, Any] = {"optimizer": use_optimizer, "agents": {}}

    if use_optimizer == "none":
        # Produce heuristic prompt templates per agent
        for agent, exs in data.items():
            if not isinstance(exs, list):
                continue
            prompt = _heuristic_prompt_for_agent(str(agent), exs)
            avg_len = 0.0
            try:
                outs = [e.get("output") for e in exs if isinstance(e, dict)]
                lens = [len(str(o)) for o in outs]
                avg_len = (sum(lens) / len(lens)) if lens else 0.0
            except Exception:
                avg_len = 0.0
            results["agents"][str(agent)] = {
                "prompt_template": prompt,
                "metrics": {"examples": len(exs), "avg_output_len": avg_len},
            }
    else:
        try:
            import dspy  # type: ignore[import-not-found]
            from dspy.teleprompt import BootstrapFewShot, MIPROv2  # type: ignore

            # Configure a local LM if the environment defines one; otherwise DSPy may default.
            # Users can set dspy globally or via env; keep this minimal to avoid secrets here.
            # dspy.settings.configure(lm=dspy.LM("openai/gpt-4o-mini"))  # optional

            for agent, exs in data.items():
                if not isinstance(exs, list) or not exs:
                    continue
                # Build DSPy examples
                dspy_examples: list[Any] = []
                for ex in exs:
                    if not isinstance(ex, dict):
                        continue
                    inp = ex.get("inputs", {}) if isinstance(ex.get("inputs"), dict) else {}
                    mem = ex.get("memory", {}) if isinstance(ex.get("memory"), dict) else {}
                    out = ex.get("output")
                    # Coerce memory into a stringy field to keep the signature simple
                    # while passing along structured content.
                    example = dspy.Example(messages=inp.get("messages"), memory=mem, output=out)
                    example = example.with_inputs("messages", "memory")
                    dspy_examples.append(example)

                signature = dspy.Signature("messages, memory -> output")
                program = dspy.ChainOfThought(signature)

                if use_optimizer == "bootstrap":
                    optimizer = BootstrapFewShot(metric=(lambda x, y, trace=None: True))
                elif use_optimizer == "mipro":
                    optimizer = MIPROv2(metric=(lambda x, y, trace=None: True), auto="light")
                else:
                    optimizer = BootstrapFewShot(metric=(lambda x, y, trace=None: True))

                compiled = optimizer.compile(program, trainset=dspy_examples)
                # Save a lightweight spec for the agent
                # We avoid serializing any LM keys or large state; just module config.
                try:
                    spec = getattr(compiled, "signature", None)
                except Exception:
                    spec = None
                results["agents"][str(agent)] = {
                    "prompt_module": "ChainOfThought(messages, memory -> output)",
                    "optimizer": use_optimizer,
                    "metrics": {"examples": len(dspy_examples)},
                    **({"signature": str(spec)} if spec else {}),
                }
        except Exception as exc:
            # Fallback to heuristics if DSPy import or compile fails
            results["optimizer_error"] = str(exc)
            for agent, exs in data.items():
                if not isinstance(exs, list):
                    continue
                prompt = _heuristic_prompt_for_agent(str(agent), exs)
                results["agents"][str(agent)] = {
                    "prompt_template": prompt,
                    "metrics": {"examples": len(exs)},
                }

    # Optional: emit overrides mapping directly consumable by `dspy fork`
    if bool(getattr(args, "emit_overrides", False)):
        overrides: dict[str, Any] = {}
        for agent, exs in data.items():
            try:
                # Prefer an explicit prompt_template from the results when present
                val = (
                    results.get("agents", {}).get(agent, {})
                    if isinstance(results.get("agents"), dict)
                    else {}
                )
                pt = val.get("prompt_template") if isinstance(val, dict) else None
                if isinstance(pt, str) and pt:
                    overrides[str(agent)] = pt
                else:
                    # Fallback to heuristic prompt construction from dataset examples
                    if isinstance(exs, list):
                        overrides[str(agent)] = _heuristic_prompt_for_agent(str(agent), exs)
            except Exception:
                continue
        out = getattr(args, "out", None)
        if out:
            Path(out).write_text(dumps_text(overrides), encoding="utf-8")
            print(f"Wrote overrides to {out}")
            return 0
        print_json(overrides)
        return 0

    out = getattr(args, "out", None)
    if out:
        Path(out).write_text(dumps_text(results), encoding="utf-8")
        print(f"Wrote prompts to {out}")
        return 0
    print_json(results)
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    dspy = sub.add_parser("dspy", help="Build DSPy datasets or optimize prompts")
    dsub = dspy.add_subparsers(dest="mode", required=True)

    bld = dsub.add_parser("build-dataset", help="Build per-agent DSPy dataset from a run")
    bld.add_argument("run_id", help="Run ID to export as dataset")
    bld.add_argument("--agents", dest="agents", default=None, help="Comma-separated agent list")
    bld.add_argument("--out", dest="out", default=None, help="Path to write dataset JSON")
    bld.set_defaults(func=_handler_build)

    opt = dsub.add_parser("optimize", help="Optimize prompts from a built dataset")
    opt.add_argument("dataset", help="Path to dataset JSON built by build-dataset")
    opt.add_argument(
        "--optimizer",
        dest="optimizer",
        default="none",
        choices=["none", "bootstrap", "mipro"],
        help="Optimizer to use (requires DSPy for bootstrap/mipro)",
    )
    opt.add_argument(
        "--emit-overrides",
        dest="emit_overrides",
        action="store_true",
        help="Emit agent->override JSON directly usable by 'dspy fork'",
    )
    opt.add_argument("--out", dest="out", default=None, help="Path to write prompts JSON")
    opt.set_defaults(func=_handler_optimize)

    # --- fork with prompt overrides ---
    def _handler_fork(args: argparse.Namespace, store: LocalStore) -> int:
        # Load overrides spec (JSON)

        from ...bindings import bind_langgraph_playback
        from ...events import Run as _Run
        from ...replay import LangGraphReplayer

        try:
            prompt_overrides = load_prompt_overrides(Path(args.overrides))
        except Exception as exc:
            print("Failed to read overrides:", exc)
            return 1

        # Import app factory
        try:
            factory = load_factory(args.app_factory)
            graph = factory()
        except Exception as exc:  # pragma: no cover
            print("Failed to import app factory:", exc)
            return 1

        teardowns: list[Any] = []

        def installer(llm: Any, tool: Any, memory: Any) -> None:
            try:
                # Honor strict_meta on wrappers
                try:
                    llm.strict_meta = bool(args.strict_meta)
                    tool.strict_meta = bool(args.strict_meta)
                    # Allow prompt hash diffs per flag (defaults to False)
                    llm.allow_diff = bool(getattr(args, "allow_diff", False))
                    # Thread prompt overrides explicitly
                    llm.prompt_overrides = dict(prompt_overrides)
                except Exception:
                    pass
                td = bind_langgraph_playback(
                    graph=graph,
                    llm=llm,
                    tool=tool,
                    memory=memory,
                    prompt_overrides=prompt_overrides,
                )
                teardowns.append(td)
            except Exception as exc:  # pragma: no cover
                print("Warning: failed to bind playback wrappers:", exc)

        replayer = LangGraphReplayer(graph=graph, store=store)
        new_id = replayer.fork_with_prompt_overrides(
            UUID(args.run_id),
            prompt_overrides,
            args.thread_id,
            install_wrappers=installer,
            freeze_time=bool(getattr(args, "freeze_time", False)),
            allow_diff=bool(getattr(args, "allow_diff", False)),
        )

        # Optionally execute and record the fork
        if bool(getattr(args, "record_fork", False)):
            try:
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
                    labels={"branch_of": str(args.run_id), "override_step": "prompt_overrides"},
                )
                # Begin a recording session to enable staged hashes from overrides
                from ...bindings import begin_recording_session
                from ...langgraph import LangGraphRecorder as _LGRecorder

                end_session = begin_recording_session(new_run.run_id)
                try:
                    store.create_run(new_run)
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
                finally:
                    try:
                        end_session()
                    except Exception:
                        pass
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
            print("Fork prepared with prompt overrides")
            print("New run id:", new_id)
            print(
                "Note: to record the fork immediately, re-run with --record-fork or "
                "run your app with the recorder."
            )
            return 0

    fork = dsub.add_parser("fork", help="Fork a run applying per-agent prompt overrides")
    fork.add_argument("run_id", help="Base run ID to fork")
    fork.add_argument("--app", dest="app_factory", required=True, help="module:function factory")
    fork.add_argument("--overrides", dest="overrides", required=True, help="Path to overrides JSON")
    fork.add_argument("--thread", dest="thread_id", default=None, help="LangGraph thread id")
    fork.add_argument(
        "--allow-diff",
        dest="allow_diff",
        action="store_true",
        help="Allow prompt hash diffs on replay",
    )
    fork.add_argument(
        "--strict-meta",
        dest="strict_meta",
        action="store_true",
        help="Enforce model meta validation",
    )
    fork.add_argument(
        "--freeze-time", dest="freeze_time", action="store_true", help="Freeze time during replay"
    )
    fork.add_argument(
        "--record-fork",
        dest="record_fork",
        action="store_true",
        help="Execute and record the fork immediately",
    )
    fork.set_defaults(func=_handler_fork)
