# Human-in-the-Loop (HITL) Patterns with LangGraph

This guide shows how Timewarp records and replays HITL flows in LangGraph, and how to configure your recorder and CLI to preserve human decisions deterministically.

Key points
- Record decisions: Timewarp emits DECISION events when the values stream changes `next` routing. Use these anchors to navigate HITL pauses and choices.
- Preserve order: Replay serializes concurrency using the recorded completion order, so the same post-HITL state is reconstructed.
- No network egress on replay: LLM/tool calls are injected from recorded events; human responses recorded in the timeline are preserved.

Recommended recorder settings
- Enable the `values` stream to expose `next` changes (for DECISION events), and optionally `updates`/`messages` for richer context.
- Take snapshots at decision boundaries to accelerate `goto` and state inspection.

Python (recorder)
```python
from timewarp.langgraph import LangGraphRecorder
from timewarp.events import Run
from timewarp.store import LocalStore
from timewarp.pruners import messages_pruner

store = LocalStore(db_path=Path("./timewarp.db"), blobs_root=Path("./blobs"))
run = Run(project="demo", name="hitl", framework="langgraph")

rec = LangGraphRecorder(
    graph=compiled_graph,
    store=store,
    run=run,
    stream_modes=("values", "updates", "messages"),
    stream_subgraphs=True,
    snapshot_on={"terminal", "decision"},
    state_pruner=messages_pruner(max_len=2000, max_items=200),
)
_ = rec.invoke(inputs, config={"configurable": {"thread_id": "t-1"}})
```

LangGraph HITL reference
- Use a checkpointer (`compile(checkpointer=...)`) so pauses/interrupts persist.
- Adopt patterns from LangGraph HITL docs (interrupt calls, human routing). Decisions will surface as `DECISION` events and state changes in `values`.

Replay workflow
- Resume deterministically at or after a human decision step:

```bash
timewarp ./timewarp.db ./blobs resume <run_id> --from 42 --thread t-1 --app path.to:make_graph
```

- In the debugger REPL:
  - `list type=DECISION thread=t-1` to view decisions
  - `goto <step>` to jump to a decision
  - `state --pretty` to inspect the state the human saw/chose

Notes
- If your HITL pattern emits explicit tool calls or messages with human content, they will be logged as TOOL/LLM events and preserved on replay.
- DECISION events include an `anchor_id` for diffing/alignment across runs with different routing.
- For sensitive human inputs, combine with `privacy_marks` (see privacy.md).
