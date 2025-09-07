from __future__ import annotations

import asyncio
import operator

# Full multi-agent LangGraph demo wired for Timewarp debugging.
#
# This example builds a representative multi-agent workflow with:
# - LLM calls (recorded as LLM events, with prompt hashes)
# - Tool calls (recorded as TOOL events, with args hashes and MCP-like metadata)
# - Routing decisions (DECISION anchors via values.next)
# - Human-in-the-loop interrupts (HITL)
# - Periodic and decision snapshots (SNAPSHOT)
#
# It is designed to exercise Timewarp's debugger: list, filters, diffs, resume,
# and injection. The graph avoids any real network calls by using a fake chat
# model and a local dummy tool.
#
# Run directly to record a run and demonstrate a resume + what-if injection:
#   uv run python -m examples.langgraph_demo.multi_agent_full
# Then explore via CLI:
#   uv run timewarp ./timewarp.sqlite3 ./blobs list
#   uv run timewarp ./timewarp.sqlite3 ./blobs debug <run_id>
# Programmatic resume is also shown in __main__.
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal


def make_graph_multi(*, include_async: bool = False) -> Any:
    try:
        from typing_extensions import TypedDict
    except Exception:  # pragma: no cover - older Python fallback
        from typing import TypedDict  # type: ignore[no-redef]

    try:
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import Command, interrupt
    except Exception as exc:  # pragma: no cover - optional dep
        raise RuntimeError("LangGraph is required for the multi-agent demo app") from exc

    # State carries message-like content and working fields
    class State(TypedDict, total=False):
        # Keep a message-like key to exercise pruners and pretty state views
        messages: list[dict[str, Any]]
        # Working fields across agents
        plan: str
        draft: str
        code_result: int
        async_result: int
        report: str
        # Track for human node routing (optional)
        last_active_agent: str
        # Tool call observational metadata (to enable TOOL event classification)
        tool_name: str
        tool_kind: str
        mcp_server: str
        mcp_transport: str
        args: list[Any]
        kwargs: dict[str, Any]
        # Aggregate artifacts across parallel branches (use reducer to avoid conflicts)
        artifacts: Annotated[list[str], operator.add]

    # Optional Fake chat model; we also embed LLM context into the update payload so
    # the recorder classifies the update as an LLM event even without messages stream.
    fake_llm: Any | None
    try:  # pragma: no cover
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        fake_llm = FakeListChatModel(
            responses=[
                "Here's a concise plan.",
                "Draft created.",
                "Approved by human.",
            ]
        )
    except Exception:  # pragma: no cover - avoid hard dep
        fake_llm = None

    # A small dummy MCP-like tool with recognizable attributes
    @dataclass
    class DummyMCPTool:
        tool_name: str = "calc"
        mcp_server: str = "stdio://dummy"
        mcp_transport: str = "stdio"

        def __call__(self, a: int, b: int, *, op: str = "add", ssn: str | None = None) -> int:
            if op == "mul":
                return a * b
            return a + b

    dummy_tool = DummyMCPTool()

    def triage(state: State) -> dict[str, Any]:
        # Seed a messages list if absent so snapshots show message-like data
        msgs = list(state.get("messages", []))
        msgs.append({"role": "system", "content": "triage -> plan + tool in parallel"})
        return {"messages": msgs}

    def planner_llm(state: State) -> dict[str, Any]:
        # Drive a fake LLM and embed a messages-like envelope for the recorder
        user_prompt = "Create a plan for the task"
        ai_text = "Here's a concise plan."
        try:
            if fake_llm is not None:
                _ = fake_llm.invoke(user_prompt)  # stage prompt hash via record taps
        except Exception:
            pass
        return {
            # Mark this update as LLM by including input/outputs in metadata-like keys
            "llm_input_messages": [
                {"role": "system", "content": "You are a planner"},
                {"role": "user", "content": user_prompt},
            ],
            "message": {"role": "assistant", "content": ai_text},
            "metadata": {
                "provider": "fake",
                "model": "fake-chat",
                "params": {"temperature": 0.0},
            },
            # Also persist typed state for downstream
            "plan": ai_text,
        }

    def tooling(state: State) -> dict[str, Any]:
        # Call a dummy MCP tool; package call context so recorder classifies as TOOL
        a, b = 6, 7
        ssn = "123-45-6789"  # intentionally sensitive; will be redacted via privacy marks
        res = dummy_tool(a, b, op="mul", ssn=ssn)
        return {
            # Keys recognized by recorder for TOOL classification + args hashing
            "tool": dummy_tool,
            "tool_name": dummy_tool.tool_name,
            "tool_kind": "MCP",
            "mcp_server": dummy_tool.mcp_server,
            "mcp_transport": dummy_tool.mcp_transport,
            "args": [a, b],
            "kwargs": {"op": "mul", "ssn": ssn},
            # Save the result to state
            "code_result": int(res),
            "artifacts": ["code"],
        }

    async def tooling_async(state: State) -> dict[str, Any]:
        # Simulate async workload (e.g., non-blocking I/O)
        await asyncio.sleep(0.01)
        a, b = 2, 5
        res = a + b
        return {"async_result": int(res), "artifacts": ["async"]}

    def human_review(state: State) -> Command[Literal["review", "compose"]]:
        # Request human input using official interrupt primitive
        decision = interrupt(
            {
                "question": "Approve generated plan and results?",
                "plan": state.get("plan", "<no plan>"),
                "result": state.get("code_result", 0),
            }
        )
        # Normalize to a boolean-like decision
        if str(decision).strip().lower() in {"y", "yes", "approve", "approved", "true"}:
            return Command(goto="review", update={"last_active_agent": "human"})
        return Command(goto="compose", update={"last_active_agent": "human"})

    def compose(state: State) -> dict[str, Any]:
        plan = state.get("plan", "<no plan>")
        code = state.get("code_result", 0)
        async_res = state.get("async_result", 0)
        draft = state.get("draft", "<no draft>")
        artifacts = (
            ",".join(state.get("artifacts", [])) if isinstance(state.get("artifacts"), list) else ""
        )
        report = (
            f"Plan={plan} | result={code}+{async_res} | draft={draft} | artifacts=[{artifacts}]"
        )
        # Also add a message for pretty state
        msgs = list(state.get("messages", []))
        msgs.append({"role": "assistant", "content": report})
        return {"report": report, "messages": msgs}

    # Subgraph: lightweight "review" pipeline that produces a draft, then light edit
    class ReviewState(TypedDict, total=False):
        messages: list[dict[str, Any]]
        draft: str

    def draft_writer(state: ReviewState) -> dict[str, Any]:
        # Best-effort LLM-marked node
        user_prompt = "Draft a short report section"
        ai_text = "Draft created."
        try:
            if fake_llm is not None:
                _ = fake_llm.invoke(user_prompt)
        except Exception:
            pass
        return {
            "llm_input_messages": [
                {"role": "system", "content": "You are a writer"},
                {"role": "user", "content": user_prompt},
            ],
            "message": {"role": "assistant", "content": ai_text},
            "metadata": {"provider": "fake", "model": "fake-chat", "params": {"temperature": 0.0}},
            "draft": ai_text,
        }

    def light_edit(state: ReviewState) -> dict[str, Any]:
        d = state.get("draft", "")
        return {"draft": d + " (edited)"}

    # Build a subgraph for review
    review = StateGraph(ReviewState)
    review.add_node("draft_writer", draft_writer)
    review.add_node("light_edit", light_edit)
    review.add_edge(START, "draft_writer")
    review.add_edge("draft_writer", "light_edit")
    review_compiled = review.compile()

    g = StateGraph(State)
    g.add_node("triage", triage)
    g.add_node("planner", planner_llm)
    g.add_node("tooling", tooling)
    if include_async:
        g.add_node("tooling_async", tooling_async)
    g.add_node("human", human_review)
    g.add_node("review", review_compiled)  # subgraph node
    g.add_node("compose", compose)
    # Wire graph: fan-out (planner + tooling + async),
    # gate through human, subgraph, then compose
    g.add_edge(START, "triage")
    g.add_edge("triage", "planner")
    g.add_edge("triage", "tooling")
    if include_async:
        g.add_edge("triage", "tooling_async")
    # Join through a human gate to emit HITL
    g.add_edge("planner", "human")
    g.add_edge("tooling", "human")
    if include_async:
        g.add_edge("tooling_async", "human")
    # Subgraph step (streams when subgraphs=True)
    g.add_edge("human", "review")
    # Allow dynamic routing to skip review
    g.add_edge("human", "compose")
    g.add_edge("review", "compose")
    g.add_edge("compose", END)

    saver = InMemorySaver()
    compiled = g.compile(checkpointer=saver)
    return compiled


def _record_run(graph: Any) -> tuple[Any, Any]:
    """Record a run using Timewarp's wrap() facade; return (recorder, result)."""
    from timewarp import messages_pruner, wrap

    rec = wrap(
        graph,
        project="demo",
        name="multi-agent-debug",
        snapshot_every=5,
        snapshot_on=("terminal", "decision"),
        stream_modes=("updates", "messages", "values"),
        state_pruner=messages_pruner(max_len=500, max_items=50),
        enable_record_taps=True,
        event_batch_size=20,
        privacy_marks={
            # Redact sensitive tool kwargs
            "kwargs.ssn": "mask4",
        },
    )
    # Provide thread_id so checkpoints are labeled; durability defaults to sync when thread present
    cfg: dict[str, Any] = {"configurable": {"thread_id": "t-demo"}}
    result = rec.invoke({"messages": [{"role": "user", "content": "do work"}]}, config=cfg)
    return rec, result


def _resume_from_checkpoint(run_id: Any) -> None:
    """Demonstrate Replay.resume programmatically with playback wrappers bound."""
    from pathlib import Path

    from timewarp.replay import Replay
    from timewarp.store import LocalStore

    store = LocalStore(db_path=Path("timewarp.sqlite3"), blobs_root=Path("blobs"))
    # Use our own factory path so Replay can import the app
    factory = "examples.langgraph_demo.multi_agent_full:make_graph_multi"
    _ = Replay.resume(
        store,
        app_factory=factory,
        run_id=run_id,
        from_step=None,
        thread_id="t-demo",
        strict_meta=False,
        freeze_time=True,
    )


def _inject_what_if(run_id: Any) -> Any:
    """Prepare and record a one-off what-if: override the first TOOL output.

    Returns the new forked run id.
    """
    from pathlib import Path
    from uuid import UUID

    from timewarp.events import Run
    from timewarp.langgraph import LangGraphRecorder
    from timewarp.replay import LangGraphReplayer
    from timewarp.store import LocalStore

    store = LocalStore(db_path=Path("timewarp.sqlite3"), blobs_root=Path("blobs"))
    events = store.list_events(UUID(str(run_id)))
    # Prefer a TOOL step; else the first LLM step
    target_step: int | None = next((e.step for e in events if e.action_type.value == "TOOL"), None)
    if target_step is None:
        target_step = next((e.step for e in events if e.action_type.value == "LLM"), None)
    if target_step is None:
        return None

    # Build graph and prepare override using the replayer
    graph = make_graph_multi()
    replayer = LangGraphReplayer(graph=graph, store=store)

    teardowns: list[Callable[[], None]] = []

    def installer(llm: Any, tool: Any, memory: Any) -> None:
        from timewarp.bindings import bind_langgraph_playback

        td = bind_langgraph_playback(graph, llm, tool, memory)
        teardowns.append(td)

    # For LLM events, a minimal replacement message is sufficient
    alt_output = {"message": {"content": "[what-if override]"}}
    new_id = replayer.fork_with_injection(
        UUID(str(run_id)),
        target_step,
        alt_output,
        thread_id="t-demo",
        install_wrappers=installer,
        freeze_time=True,
    )

    try:
        # Recover original input payload
        orig_input: Any | None = None
        for e in events:
            if e.input_ref is not None:
                orig_input = __import__("timewarp.codec", fromlist=["from_bytes"]).from_bytes(
                    store.get_blob(e.input_ref)
                )
                break
        if orig_input is None:
            return new_id
        # Record the forked execution under the provided run id
        new_run = Run(
            run_id=new_id,
            project="demo",
            name="multi-agent-debug-fork",
            framework="langgraph",
            labels={"branch_of": str(run_id), "override_step": str(target_step)},
        )
        rec = LangGraphRecorder(
            graph=graph,
            store=store,
            run=new_run,
            stream_modes=("updates", "messages", "values"),
            stream_subgraphs=True,
            snapshot_every=5,
            snapshot_on={"terminal", "decision"},
        )
        _ = rec.invoke(orig_input, config={"configurable": {"thread_id": "t-demo"}})
        return new_id
    finally:
        for td in teardowns:
            try:
                td()
            except Exception:
                pass


if __name__ == "__main__":
    # Build and record a sync run (graph without async nodes)
    graph = make_graph_multi(include_async=False)
    recorder, result_state = _record_run(graph)
    print("Recorded run_id:", recorder.last_run_id)
    # Programmatic resume using recorded outputs (no network/tool side effects)
    _resume_from_checkpoint(recorder.last_run_id)
    # Prepare + record a what-if fork by overriding the first TOOL output
    fork_id = _inject_what_if(recorder.last_run_id)
    if fork_id is not None:
        print("Forked run recorded:", fork_id)
    else:
        print("No TOOL event found; skip fork")

    # Record an async run to exercise astream path
    async def _run_async() -> None:
        from timewarp import messages_pruner, wrap

        graph_async = make_graph_multi(include_async=True)
        rec = wrap(
            graph_async,
            project="demo",
            name="multi-agent-debug-async",
            snapshot_every=5,
            snapshot_on=("terminal", "decision"),
            stream_modes=("updates", "messages", "values"),
            state_pruner=messages_pruner(max_len=500, max_items=50),
            enable_record_taps=True,
            event_batch_size=20,
        )
        cfg: dict[str, Any] = {"configurable": {"thread_id": "t-demo-async"}}
        _ = await rec.ainvoke({"messages": [{"role": "user", "content": "do work"}]}, config=cfg)
        print("Recorded async run_id:", rec.last_run_id)

    try:
        asyncio.run(_run_async())
    except Exception:
        # Async path depends on graph.astream support; skip if unavailable
        print("Async astream path unavailable; skipped")
