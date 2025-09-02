Timewarp — Deterministic Replay & Time‑Travel Debugger for LLM Agent Workflows
==============================================================================

Record every step. Rewind any step. Reproduce any run.

Timewarp adds event‑sourced logging and deterministic replay to agent frameworks (LangGraph first, LangChain optional), plus a CLI debugger for step‑through, diffs, and what‑if edits. It fills a well‑documented gap: mainstream tools visualize traces but don’t let you replay them exactly.

What’s Included (v0.1 core)
---------------------------

- Core models and helpers
  - `timewarp.events`: Pydantic v2 models (`Run`, `Event`, `BlobRef`), hashing, redaction
  - `timewarp.codec`: Canonical JSON (orjson), Zstandard compression
  - `timewarp.determinism`: RNG snapshot/restore
- Local store
  - `timewarp.store.LocalStore`: SQLite (WAL) for runs/events + filesystem blobs
  - Deterministic blob layout: `runs/<run_id>/events/<step>/<kind>.bin` (zstd)
  - Connection PRAGMAs applied per-connection: `journal_mode=WAL`, `synchronous=NORMAL`, configurable busy timeout
- LangGraph recording adapter
  - `timewarp.adapters.langgraph.LangGraphRecorder`: streams `updates|values|messages`, records `LLM|TOOL|DECISION|HITL|SNAPSHOT` events
  - Labels include `thread_id`, `namespace`, `node`, `checkpoint_id`, `anchor_id`
  - Privacy redaction via `privacy_marks`
- Diff engine
  - Anchor‑aware alignment + windowed realignment; DeepDiff/text diffs; first divergence
- Replay scaffolding
  - `PlaybackLLM`/`PlaybackTool` inject recorded outputs with prompt/args validation
  - `LangGraphReplayer.resume()` re‑executes from nearest checkpoint using recorded outputs
  - What‑if overrides supported (one‑shot per step)
- CLI
  - `timewarp list|debug|diff`, plus `resume` and `inject` commands (see below)
  - `export langsmith <run_id>` to serialize runs/events for external tooling
- Telemetry (optional)
  - OpenTelemetry spans per event; replay spans link to originals via Span Links
  - Attributes use `tw.*` keys: `tw.run_id`, `tw.step`, `tw.action_type`, `tw.actor`, `tw.replay`,
    `tw.namespace`, `tw.thread_id`, `tw.checkpoint_id`, `tw.anchor_id`, `tw.branch_of`,
    `tw.hash.output|state|prompt`

Install & Dev
-------------

Requires Python 3.12+.

```
uv venv && uv pip install -e .[dev]
ruff format && ruff check --fix
mypy --strict src
pytest -q
```

Optional dependencies
- Adapters: `uv pip install -e .[adapters]` (installs `langgraph`, `langchain-core`)
- Telemetry: `uv pip install -e .[otel]` (installs `opentelemetry-*`)

Recording a Run (LangGraph)
---------------------------

Quickstart via facade:

```
from timewarp import wrap, messages_pruner
from examples.langgraph_demo.app import make_graph

graph = make_graph()

rec = wrap(
    graph,
    project="demo",
    name="my-run",
    stream_modes=("updates", "messages", "values"),
    snapshot_every=20,
    snapshot_on=("terminal", "decision"),
    state_pruner=messages_pruner(max_len=2000, max_items=200),
    enable_record_taps=True,  # robust prompt/tool args hashing
    event_batch_size=20,      # batch appends to reduce SQLite overhead
)
result = rec.invoke({"text": "hi"}, config={"configurable": {"thread_id": "t-1"}})
print("run_id=", rec.last_run_id)
```

Manual recorder usage:

```
from pathlib import Path
from timewarp.events import Run
from timewarp.store import LocalStore
from timewarp.adapters.langgraph import LangGraphRecorder
from timewarp import messages_pruner

store = LocalStore(db_path=Path("./timewarp.db"), blobs_root=Path("./blobs"))
run = Run(project="demo", name="my-run", framework="langgraph")
rec = LangGraphRecorder(
    graph=my_compiled_graph,
    store=store,
    run=run,
    stream_modes=("updates", "values"),  # also supports "messages"
    stream_subgraphs=True,
    snapshot_on={"terminal", "decision"},
    state_pruner=messages_pruner(max_len=2000, max_items=200),
)
result = rec.invoke({"text": "hi"}, config={"configurable": {"thread_id": "t-1"}})
```

Debugging & Diffs
-----------------

```
timewarp ./timewarp.db ./blobs list
timewarp ./timewarp.db ./blobs debug <run_id>
timewarp ./timewarp.db ./blobs diff <run_a> <run_b>
timewarp ./timewarp.db ./blobs events <run_id> --type LLM --node compose --thread t-1 --json
```

Deterministic Replay & What‑ifs (CLI)
-------------------------------------

Provide an app factory that returns a compiled LangGraph (example shipped):

```
--app examples.langgraph_demo.app:make_graph
```

Resume deterministically from a prior checkpoint:

```
timewarp ./timewarp.db ./blobs resume <run_id> --from 42 --thread t-1 --app examples.langgraph_demo.app:make_graph
```

Inject an alternative output at step N and fork:

```
timewarp ./timewarp.db ./blobs inject <run_id> 23 \
  --output alt_23.json \
  --thread t-1 \
  --app examples.langgraph_demo.app:make_graph \
  --record-fork   # execute and persist new branch immediately
```

Notes
- The CLI binds playback wrappers via lightweight installers that intercept LangChain ChatModel/Tool calls during replay. Your graph runs without network/tool side‑effects in replay mode.
- For forks, you can either prepare and record later, or pass `--record-fork` to execute and persist the new branch immediately. The new run is labeled with `branch_of` and `override_step` for lineage.
 - Snapshot knobs: `snapshot_every` controls cadence; `snapshot_on` can include `"terminal"` and/or `"decision"` to emit snapshots at run end and after routing decisions. You can also pass a `state_pruner` callable to trim large fields from state snapshots before persistence.
 - REPL filters: inside `debug`, run `list type=LLM node=compose thread=t-1` to view a subset.
 - Pretty state: `state --pretty` prints truncated previews with size hints.
 - Save patch: `savepatch STEP file.json` writes the event’s output JSON for reuse with `inject`.
  - Event batching: `event_batch_size` batches DB writes for throughput. For heavy runs, try `50` or `100`.

Record‑time taps (determinism)
------------------------------

For stronger determinism checks, Timewarp can compute and store `hashes.prompt` and `hashes.args` at call sites (LangChain core):

```
from timewarp.adapters.installers import bind_langgraph_record

teardown = bind_langgraph_record()
try:
    # run your graph under the recorder
    ...
finally:
    teardown()
```

The `wrap(...)` facade can auto‑enable taps via `enable_record_taps=True`.

Telemetry
---------

Enable OpenTelemetry by installing the extras and configuring an exporter in your app. Timewarp emits spans per event; replay spans use Span Links pointing to original spans. Attributes use the `tw.*` namespace.

Examples
--------

- Example LangGraph factory: `examples/langgraph_demo/app.py` provides `make_graph()` for quick `--app` usage in CLI.
- Freeze-time example: `examples/langgraph_demo/time_freeze_app.py` provides `make_graph_time()` that writes `timewarp.determinism.now()` into state so you can verify identical timestamps on replay with `--freeze-time`.
- Parallel branches example: `examples/langgraph_demo/parallel_app.py` demonstrates fan-out and join with DECISION anchors.
- Tests exercise recorder, diff alignment, replay state reconstruction, and playback installers.

MCP Example (optional)
----------------------

When `langgraph` and `langchain-mcp-adapters` are available, you can run the MCP demo app:

```
# Record a run using the MCP example app
python - <<'PY'
from pathlib import Path
from timewarp.store import LocalStore
from timewarp.events import Run
from timewarp.adapters.langgraph import LangGraphRecorder
from examples.langgraph_demo.mcp_app import make_graph_mcp

store = LocalStore(db_path=Path('./timewarp.db'), blobs_root=Path('./blobs'))
graph = make_graph_mcp()
run = Run(project='demo', name='mcp', framework='langgraph')
rec = LangGraphRecorder(graph=graph, store=store, run=run, stream_modes=("messages","updates"), stream_subgraphs=True)
_ = rec.invoke({"text":"hi"}, config={"configurable": {"thread_id": "t-1"}})
print('run_id=', run.run_id)
PY

# View TOOL events with MCP metadata
timewarp ./timewarp.db ./blobs events <run_id> --type TOOL --tool-kind MCP --json
```

Note: MCP metadata is best-effort and dependent on adapter/provider behavior. In environments
where the stream does not emit tool metadata, you may not observe TOOL events for MCP calls.

HITL & Privacy Docs
-------------------

- HITL patterns with LangGraph (DECISION anchors, snapshots, CLI tips): see `docs/hitl.md`.
- Privacy marks and redaction strategies (`redact`, `mask4`) with examples: see `docs/privacy.md`.

Time Provider & Freeze-Time
---------------------------

Use `timewarp.determinism.now()` in your graphs to obtain a deterministic clock.
Recording uses `now()` for `Run.started_at` and all `Event.ts`. During replay, you can
freeze time to the recorded event timestamps.

Programmatic replay:

```
from timewarp.replay import LangGraphReplayer

replayer = LangGraphReplayer(graph=my_graph, store=store)
session = replayer.resume(run_id, from_step=None, thread_id="t-1", install_wrappers=installer, freeze_time=True)
```

CLI replay with frozen time:

```
timewarp ./timewarp.db ./blobs resume <run_id> --app examples.langgraph_demo.time_freeze_app:make_graph_time --thread t-1 --freeze-time

timewarp ./timewarp.db ./blobs inject <run_id> <step> --output alt.json \
  --app examples.langgraph_demo.time_freeze_app:make_graph_time --thread t-1 --freeze-time --record-fork
```

Example graph writes the ISO timestamp to state (key `now_iso`). With `--freeze-time`,
replay preserves the exact value that was recorded.

Replay Convenience Facade
-------------------------

You can also resume deterministically via a one-call facade:

```
from timewarp import Replay

session = Replay.resume(
    store,
    app_factory="examples.langgraph_demo.app:make_graph",
    run_id=<UUID>,
    from_step=42,
    thread_id="t-1",
    strict_meta=True,
    freeze_time=True,
)
print(session.result)
```

Exporters
---------

Use the CLI to export a run in a LangSmith-friendly JSON bundle:

```
timewarp ./timewarp.db ./blobs export langsmith <run_id> --include-blobs
```

The module `timewarp.exporters.langsmith` also exposes `serialize_run(...)` and `export_run(...)` for programmatic use.

OpenTelemetry Quickstart
------------------------

See `docs/otel-quickstart.md` for a minimal setup to emit spans per event and link replay spans to recorded ones.
