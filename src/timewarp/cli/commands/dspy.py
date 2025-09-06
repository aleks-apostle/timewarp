from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from uuid import UUID

from ...exporters.dspy import build_dspy_dataset
from ...store import LocalStore
from ..helpers.jsonio import dumps_text, loads_file, print_json


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
    opt.add_argument("--out", dest="out", default=None, help="Path to write prompts JSON")
    opt.set_defaults(func=_handler_optimize)
