"""
Interactive debug REPL demo.

This script records a multi-agent LangGraph run, then opens the Timewarp
interactive debugger and drives a few commands (if `pexpect` is available).

Usage:
  uv run python -m examples.langgraph_demo.debug_repl_e2e

Notes:
  - Requires optional `langgraph` dependency for recording the example app.
  - For interactive driving, install `pexpect` (optional):
      uv pip install pexpect
  - If `pexpect` is not installed, the script will print instructions and the
    command to launch the REPL manually.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from timewarp import messages_pruner, wrap
from timewarp.bindings import stage_memory_tap
from timewarp.store import LocalStore


def _record_run(db: Path, blobs: Path) -> tuple[str, int, int | None]:
    """Record a representative run and return (run_id, llm_step, mem_step)."""
    try:
        from examples.langgraph_demo.multi_agent_full import make_graph_multi
    except Exception as exc:  # pragma: no cover - optional dep
        raise SystemExit(f"LangGraph is required: {exc}") from None

    store = LocalStore(db_path=db, blobs_root=blobs)
    graph = make_graph_multi(include_async=False)

    # Stage taps to guarantee MEMORY/RETRIEVAL presence
    stage_memory_tap(
        {
            "kind": "MEMORY",
            "mem_provider": "Mem0",
            "mem_op": "PUT",
            "key": "long.session",
            "value": {"k": "v", "n": 1},
            "mem_scope": "long",
            "mem_space": "demo",
        }
    )
    stage_memory_tap(
        {
            "kind": "RETRIEVAL",
            "mem_provider": "Mem0",
            "query": "repl-demo",
            "items": [
                {"id": "a", "text": "A", "score": 0.9},
                {"id": "b", "text": "B", "score": 0.7},
            ],
            "policy": {"retriever": "vector", "top_k": 2},
            "query_id": "qid-repl",
        }
    )

    rec = wrap(
        graph,
        project="demo",
        name="debug-repl",
        store=store,
        snapshot_every=5,
        snapshot_on=("terminal", "decision"),
        stream_modes=("updates", "messages", "values"),
        state_pruner=messages_pruner(max_len=500, max_items=50),
        enable_record_taps=True,
        enable_memory_taps=True,
        event_batch_size=20,
        privacy_marks={"kwargs.ssn": "mask4"},
    )
    cfg: dict[str, Any] = {"configurable": {"thread_id": "t-repl"}}
    _ = rec.invoke({"messages": [{"role": "user", "content": "run repl"}]}, config=cfg)
    run_id: str = str(rec.last_run_id)

    # Find a representative LLM step and MEMORY/RETRIEVAL step
    llm_step: int | None = None
    mem_step: int | None = None
    events = store.list_events(run_id)
    for e in events:
        if llm_step is None and e.action_type.value == "LLM":
            llm_step = e.step
        if mem_step is None and e.action_type.value in ("MEMORY", "RETRIEVAL"):
            mem_step = e.step
        if llm_step is not None and mem_step is not None:
            break
    return run_id, llm_step or 0, mem_step


def _spawn_debug_and_drive(
    db: Path, blobs: Path, run_id: str, llm_step: int, mem_step: int | None
) -> None:
    try:
        import pexpect  # type: ignore
    except Exception:
        print("pexpect not installed. Launch the REPL manually with:")
        print(f"  {sys.executable} -m timewarp.cli {db} {blobs} debug {run_id}")
        print("Then try commands like: list | lastllm | tools | state --pretty | prompt", llm_step)
        return

    cmd = [sys.executable, "-m", "timewarp.cli", str(db), str(blobs), "debug", run_id]
    child = pexpect.spawn(" ".join(cmd), timeout=10, encoding="utf-8")

    def expect_prompt() -> None:
        child.expect_exact("> ")

    # Header
    child.expect("schema=")
    child.expect("Commands:")
    expect_prompt()

    # List timeline
    child.sendline("list")
    expect_prompt()

    # Show last LLM and tools summary
    child.sendline("lastllm")
    expect_prompt()
    child.sendline("tools")
    expect_prompt()

    # Prompt + tokens for the LLM step
    child.sendline(f"prompt {llm_step}")
    expect_prompt()
    child.sendline(f"tokens {llm_step}")
    expect_prompt()

    # Memory views if present
    if mem_step is not None:
        child.sendline("memory")
        expect_prompt()
        child.sendline(f"memory show {mem_step}")
        expect_prompt()

    # Pretty state
    child.sendline("state --pretty")
    expect_prompt()

    # Save a patch for the LLM step
    out_dir = Path("tmp_run")
    out_dir.mkdir(parents=True, exist_ok=True)
    patch_path = out_dir / f"alt_{llm_step}.json"
    child.sendline(f"savepatch {llm_step} {patch_path}")
    expect_prompt()

    # Quit
    child.sendline("quit")
    child.expect(pexpect.EOF)
    if patch_path.exists():
        print(f"Saved patch to {patch_path}")


def main() -> None:
    root = Path.cwd()
    tw_root = root / "tw_runs"
    tw_root.mkdir(exist_ok=True)
    db = tw_root / "demo.sqlite3"
    blobs = tw_root / "blobs"
    blobs.mkdir(exist_ok=True)

    run_id, llm_step, mem_step = _record_run(db, blobs)
    print("Recorded run_id:", run_id)

    # Drive the REPL if possible; otherwise print instructions
    _spawn_debug_and_drive(db, blobs, run_id, llm_step, mem_step)


if __name__ == "__main__":
    main()
