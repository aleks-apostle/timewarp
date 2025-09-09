Timewarp — Deterministic Replay & Time‑Travel Debugger for LLM Agent Workflows
==============================================================================

[![PyPI](https://img.shields.io/pypi/v/timewarp-llm.svg)](https://pypi.org/project/timewarp-llm/)
[![Python Versions](https://img.shields.io/pypi/pyversions/timewarp-llm.svg)](https://pypi.org/project/timewarp-llm/)
[![License](https://img.shields.io/pypi/l/timewarp-llm.svg)](./LICENSE)
[![CI](https://github.com/aleks-apostle/timewarp/actions/workflows/ci.yml/badge.svg)](https://github.com/aleks-apostle/timewarp/actions/workflows/ci.yml)
[![Publish](https://github.com/aleks-apostle/timewarp/actions/workflows/release.yml/badge.svg)](https://github.com/aleks-apostle/timewarp/actions/workflows/release.yml)

Record every step. Rewind any step. Reproduce any run.

Timewarp adds event‑sourced logging and deterministic replay to agent frameworks (LangGraph first, LangChain optional), plus an interactive REPL debugger for step‑through, diffs, and what‑if edits. It fills a well‑documented gap: mainstream tools visualize traces but don’t let you replay them exactly.

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
  - Monotonic steps: per-run event `step` must strictly increase; single-writer-per-run is recommended for correctness
- LangGraph recording
  - `timewarp.langgraph.LangGraphRecorder`: streams `updates|values|messages`, records `LLM|TOOL|DECISION|HITL|SNAPSHOT` events
  - Labels include `thread_id`, `namespace`, `node`, `checkpoint_id`, `anchor_id`
  - Privacy redaction via `privacy_marks`
- Diff engine
  - Anchor‑aware alignment + windowed realignment; DeepDiff/text diffs; first divergence
  - Delta debugging: minimal failing window via `diff --bisect`
- Replay scaffolding
  - `PlaybackLLM`/`PlaybackTool` inject recorded outputs with prompt/args validation
  - `LangGraphReplayer.resume()` re‑executes from nearest checkpoint using recorded outputs
  - What‑if overrides supported (one‑shot per step)
- CLI
  - `timewarp list|events|tools|diff|debug`, plus `resume`, `inject`, and `fsck` (see below)
  - `timewarp-repl` interactive debugger for browsing timelines, inspecting prompts/tools/memory, deterministic resume, and recording what‑if forks
  - `export langsmith <run_id>` to serialize runs/events for external tooling
- Telemetry (optional)
  - OpenTelemetry spans per event; replay spans link to originals via Span Links
  - Attributes use `tw.*` keys: `tw.run_id`, `tw.step`, `tw.action_type`, `tw.actor`, `tw.replay`,
    `tw.namespace`, `tw.thread_id`, `tw.checkpoint_id`, `tw.anchor_id`, `tw.branch_of`,
    `tw.hash.output|state|prompt`

Install & Dev
-------------

Requires Python 3.11+.

Install from PyPI:

```
pip install timewarp-llm
# optional extras
pip install langgraph langchain-core  # optional runtime dependencies for recording/replay
pip install 'timewarp-llm[otel]'
pip install 'timewarp-llm[dspy]'   # optional DSPy optimizers

# CLI entry points
timewarp --help
timewarp-repl --help
```

```
uv venv && uv pip install -e .[dev]
ruff format && ruff check --fix
mypy --strict
pytest -q

Developer notes: see `docs/DEV.md` for CLI helper modules, canonical JSON path, store insert/observability details, and provenance consistency.
```

Optional dependencies
- LangGraph/LC core: `uv pip install langgraph langchain-core`
- Telemetry: `uv pip install -e .[otel]` (installs `opentelemetry-*`)
- DSPy: `uv pip install -e .[dspy]` (installs `dspy`) for optional prompt optimization

DSPy Dataset & Optimization (optional)
--------------------------------------

Timewarp can export per-agent datasets from recorded runs and optionally run DSPy optimizers
to produce improved prompt specifications.

Build a dataset from a run:

```
timewarp ./timewarp.sqlite3 ./blobs dspy build-dataset <run_id> --out ds.json
```

Run an optimizer (requires installing the `dspy` extra):

```
timewarp ./timewarp.sqlite3 ./blobs dspy optimize ds.json --optimizer bootstrap --out prompts.json
# or
timewarp ./timewarp.sqlite3 ./blobs dspy optimize ds.json --optimizer mipro --out prompts.json
# emit overrides JSON directly consumable by `dspy fork`
timewarp ./timewarp.sqlite3 ./blobs dspy optimize ds.json --emit-overrides --out overrides.json
```

Notes
- Dataset groups examples by agent (LangGraph node). Each example includes inputs (messages when
  available), the agent's memory snapshot at `step-1`, the recorded output, and step/thread metadata.
- Optimizers are optional. If DSPy is not installed or compilation fails, the CLI emits a
  heuristic prompt template per agent with basic metrics.
- This is pre‑release functionality and may evolve without backward compatibility guarantees.

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
from timewarp.langgraph import LangGraphRecorder
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
timewarp ./timewarp.db ./blobs debug <run_id>              # basic inspector (legacy)
timewarp ./timewarp.db ./blobs diff <run_a> <run_b>       # first divergence / bisect
timewarp ./timewarp.db ./blobs events <run_id> --type LLM --node compose --thread t-1 --json

# NEW: interactive debugger (recommended)
timewarp-repl ./timewarp.sqlite3 ./blobs <run_id> \
  --app examples.langgraph_demo.app:make_graph \
  --thread t-1 --freeze-time
```

Interactive Debugger (REPL)
---------------------------

Timewarp ships a richer interactive REPL that unifies timeline browsing, prompt/tools/memory
inspection, deterministic replay, what‑if injections, prompt overrides, and diffs.

- Binary: `timewarp-repl` (installed by the package)
- Programmatic: `timewarp.interactive_debug.launch_debugger(...)`

CLI usage

```
timewarp-repl <db> <blobs> <run_id> [--app module:function] [--thread ID] \
  [--freeze-time] [--strict-meta] [--allow-diff] [--overrides overrides.json]
```

Inside the REPL

```
Commands
  app module:function         Load a compiled LangGraph via factory (enables resume/inject)
  thread T                    Set thread_id to use during resume/inject
  freeze | unfreeze           Toggle freeze-time replay
  strict | nonstrict          Toggle strict meta checks (provider/model/tools invariants)
  allowdiff | disallowdiff    Toggle allowing prompt diffs during replay (for overrides)
  overrides [file.json]       Load per-agent prompt overrides; empty to clear

Views
  list [type=.. node=.. thread=.. ns=..]   Timeline (filterable)
  event STEP                  Show a single event + blob size hints
  llm                         Show the last LLM before current position
  prompt [STEP]               Messages/tools head + estimated tokens for an LLM step
  tools [STEP]                Tools summary across run or details for one LLM step
  memory                      Memory summary by space
  memory_show STEP            Full memory snapshot up to step
  memory_diff A B [key=path]  Structural diff between two snapshots (optional dotted key)

Execution
  resume [FROM_STEP]          Deterministically resume from a checkpoint using recorded outputs
  inject STEP output.json [-r|--record]
                              One-shot override at STEP; optionally record fork immediately
  fork_prompts overrides.json [-r|--record]
                              Prepare/record a fork that applies prompt overrides
  diff OTHER_RUN_ID [--bisect] [--window N]
                              Show first divergence or minimal failing window
```

Programmatic launch

```
from timewarp.interactive_debug import launch_debugger
from examples.langgraph_demo.app import make_graph

launch_debugger(
    db="./timewarp.sqlite3",
    blobs="./blobs",
    run_id="<UUID>",
    graph=make_graph(),          # or pass --app module:function via CLI
    thread_id="t-1",
    freeze_time=True,
    strict_meta=False,
    allow_diff=False,
)
```

Integrity Check (fsck)
----------------------

Verify that all blobs referenced by a run exist on disk; optionally repair and garbage‑collect orphans. Emits JSON for easy automation.

```
# Basic verification (JSON output)
timewarp ./timewarp.db ./blobs fsck <run_id>
# Attempt repair by promoting any matching .tmp files to final .bin
timewarp ./timewarp.db ./blobs fsck <run_id> --repair
# Remove blob files on disk that are not referenced by the run (dangerous);
# use a grace period to avoid racing with in-flight writes
timewarp ./timewarp.db ./blobs fsck <run_id> --gc-orphans --grace 5
```

Output shape:

```
{"missing": ["runs/<id>/events/12/output.bin", ...],
 "repaired": ["runs/<id>/events/12/output.bin", ...],
"orphans_gc": ["runs/<id>/events/99/output.bin.tmp", ...]}
```

SQLite & Concurrency
--------------------

- Single-writer-per-run: Timewarp enforces strictly increasing `step` per `run_id`.
  The `events` table uses `(run_id, step)` as its primary key and `LocalStore` guards with a
  monotonic check against `MAX(step)`. Running multiple writers for the same `run_id` can cause
  UNIQUE violations or out-of-order errors. Recommended: one process per run ID.
- PRAGMAs: Each connection applies `journal_mode=WAL`, `synchronous=NORMAL`, a configurable
  `busy_timeout`, and best-effort `foreign_keys=ON`, `temp_store=MEMORY`, and a
  `journal_size_limit`. These trade-offs aim for durable, fast appends.
- JSON1 indexes (optional): Additional indexes rely on SQLite JSON1 (`json_extract`). When JSON1
  isn’t available, index creation is skipped and a one-time warning is printed; queries still work
  but may fall back to table scans. Most modern Python builds ship SQLite with JSON1 enabled.
- Blob finalization: Blobs are written to `*.bin.tmp` and promoted to final `*.bin` on event
  append. Reading a blob may also finalize the file if a matching `.tmp` exists.
- Orphan GC: `fsck --gc-orphans` applies a grace window (`--grace`) to avoid deleting files
  created moments ago by in-flight writers.

Delta Debugging (Minimal Failing Delta)
---------------------------------------

Find the smallest contiguous mismatching window between two runs, accounting for
anchor‑aware realignment to skip benign reorders:

```
# Text output
timewarp ./timewarp.db ./blobs diff <run_a> <run_b> --bisect

# JSON output (machine‑readable)
timewarp ./timewarp.db ./blobs diff <run_a> <run_b> --bisect --json
# => {"start_a": <int>, "end_a": <int>, "start_b": <int>, "end_b": <int>, "cause": <str>} | {"result": null}

# Tune anchor lookahead window (default 5)
timewarp ./timewarp.db ./blobs diff <run_a> <run_b> --bisect --window 3
```

Notes
- Causes: "output hash mismatch" | "anchor mismatch" | "adapter/schema mismatch".
- Benign reorders (by matching anchors) are excluded from the window.
- If all aligned pairs match but lengths differ, the trailing unmatched step is reported with cause "anchor mismatch".
- Exit codes: add `--fail-on-divergence` to return a non‑zero exit code when a divergence is found (applies to text and `--json` modes).

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

Tools, Prompt, and Memory Views
-------------------------------

Inspect tools available to the model, prompts, and memory/retrieval state reconstructed per step and agent:

```
# Tools summary across LLM steps
timewarp ./timewarp.sqlite3 ./blobs tools <run_id>

# Tools detail for a specific LLM step
timewarp ./timewarp.sqlite3 ./blobs tools <run_id> --step 42 --json

# Memory snapshots (per agent/space)
timewarp ./timewarp.sqlite3 ./blobs memory summary <run_id> --step 120
timewarp ./timewarp.sqlite3 ./blobs memory show <run_id> --step 120 --space planner --json
timewarp ./timewarp.sqlite3 ./blobs memory diff <run_id> 100 140 --space planner --scope working --key messages.0
Tips in the interactive REPL
> tools            # summary across LLM steps
> tools 42         # detail for step 42
> prompt 42        # prompt parts (messages + tools), hashes, token estimate
> memory           # summary by agent at current step
> memory_show 120  # snapshot at step 120
> memory_diff 120 140 key=messages.0  # structural diff; optional dot path
```

Details
- Available tools: extracted from recorded prompt parts when present; `tools_digest` shows a stable hash when details aren’t recorded.
- Called tools: correlated by `thread_id` and node, scanning forward until the next LLM event on the same thread.
- Token estimate: provider-agnostic heuristic (≈ chars/4) to quickly gauge prompt size; for precise tokens/costs integrate a tokenizer.
- Privacy: printed payloads respect `privacy_marks` redaction.

Capturing LangGraph memory from values
-------------------------------------

To synthesize memory from LangGraph `values` stream, configure the recorder with `memory_paths`:

```
from timewarp.langgraph import LangGraphRecorder

rec = LangGraphRecorder(
    graph=graph,
    store=store,
    run=run,
    stream_modes=("updates", "values"),
    memory_paths=("messages", "history", "scratch", "artifacts", "memory"),
)
```

Each new/changed key under these paths emits a MEMORY event (`mem_provider="LangGraphState"`) with stable `hashes.item`, `labels.anchor_id`, and inferred `mem_scope` from the path name. The CLI `memory` command reconstructs per-agent snapshots from these events.

Record‑time taps (determinism)
------------------------------

For stronger determinism checks, Timewarp can compute and store `hashes.prompt` and `hashes.args` at call sites (LangChain core). When using installers directly, start a recording session to scope staged hashes to the current run:

```
from timewarp.bindings import begin_recording_session, bind_langgraph_record

# Assuming you are using LangGraphRecorder with a concrete Run object
end_session = begin_recording_session(run.run_id)
teardown = bind_langgraph_record()
try:
    # run your graph under the recorder
    ...
finally:
    # Ensure both session and patches are cleaned up
    end_session()
    teardown()
```

- The `wrap(...)` facade auto‑enables record taps with `enable_record_taps=True` and管理 the session lifecycle for you.
- When not using `wrap(...)`, prefer `begin_recording_session(...)` to avoid any cross‑run leakage; global fallbacks are removed in dev.

Telemetry
---------

Enable OpenTelemetry by installing the extras and configuring an exporter in your app. Timewarp emits spans per event; replay spans use Span Links pointing to original spans. Attributes use the `tw.*` namespace.

Examples
--------

- Example LangGraph factory: `examples/langgraph_demo/app.py` provides `make_graph()` for quick `--app` usage in CLI.
- CLI implementation: the console entrypoint is `timewarp.cli:main`, which dispatches to a decomposed CLI package under `timewarp/cli/` (commands/helpers). Commands remain stable across versions.
- Freeze-time example: `examples/langgraph_demo/time_freeze_app.py` provides `make_graph_time()` that writes `timewarp.determinism.now()` into state so you can verify identical timestamps on replay with `--freeze-time`.
- Parallel branches example: `examples/langgraph_demo/parallel_app.py` demonstrates fan-out and join with DECISION anchors.

Multi‑Agent Demo
----------------

A realistic multi‑agent LangGraph with mock tools, a fake LLM, human‑in‑the‑loop (HITL), subgraphs, and dynamic routing:

- File: `examples/langgraph_demo/multi_agent_full.py`
- Features:
  - LLM events with prompt hashing via record‑time taps (FakeListChatModel)
  - TOOL events with MCP‑style metadata (`tool_name`, `mcp_server`, `mcp_transport`) and args hashing
  - Human‑in‑the‑loop using `langgraph.types.interrupt` + `Command(goto=...)`
  - Subgraph for review (`draft_writer` → `light_edit`) and dynamic routing to skip review
  - Snapshots on terminal + decisions; message‑rich state for pruning/pretty printing
- Run it to record a run and demonstrate resume + what‑if:

```
uv run python -m examples.langgraph_demo.multi_agent_full

# Inspect with CLI (defaults: ./timewarp.sqlite3 ./blobs)
uv run timewarp ./timewarp.sqlite3 ./blobs list
uv run timewarp ./timewarp.sqlite3 ./blobs debug <run_id>
```

End‑to‑End Script (Record → Resume → Fork → Diff)
------------------------------------------------

For a single, repeatable flow that exercises most features, use:

- File: `examples/langgraph_demo/run_all.py`
- What it does:
  - Builds the multi‑agent graph and records a baseline run
  - Resumes deterministically from the nearest checkpoint (no side‑effects)
  - Forks the run by overriding the first TOOL/LLM output (what‑if), records the branch
  - Computes first divergence and minimal failing window between base and fork
- Uses a dedicated store by default to avoid local schema drift:
  - DB: `tw_runs/demo.sqlite3`
  - Blobs: `tw_runs/blobs/`
- Run:

```
uv run python -m examples.langgraph_demo.run_all
# => prints JSON with run IDs, first divergence, and minimal window
```

Notes
- Ensure optional deps installed: `uv pip install langgraph langchain-core`.
- For CLI `resume`/`inject`, pass your app factory (e.g., `examples.langgraph_demo.app:make_graph` or `examples.langgraph_demo.multi_agent_full:make_graph_multi`).
- Full multi‑agent example: `examples/langgraph_demo/multi_agent_full.py` exercises LLM, TOOL,
  DECISION, HITL, SNAPSHOT, subgraphs, parallel fan‑out with reducers, and async paths.
- Tests exercise recorder, diff alignment, replay state reconstruction, and playback installers.

Full Multi‑Agent Demo
---------------------

Record a representative multi‑agent workflow and exercise the debugger end‑to‑end:

```
python -m examples.langgraph_demo.multi_agent_full
# prints: Recorded run_id: <UUID>
# also records a what‑if fork and, if supported, an async run
```

The script builds a graph with:
- LLM nodes (`planner`, `review:draft_writer`) and staged prompt hashes.
- TOOL node (`tooling`) with MCP‑like metadata and privacy redaction on kwargs.
- Parallel branches (`planner`, `tooling`, optional `tooling_async`) merged via a
  reducer on `artifacts` to avoid concurrent update conflicts.
- A `human` HITL interrupt, DECISION events on routing, and periodic/terminal snapshots.
- A `review` subgraph that streams when `stream_subgraphs=True`.

After recording, explore via CLI (defaults write to `./timewarp.sqlite3` and `./blobs`):

```
timewarp ./timewarp.sqlite3 ./blobs list
timewarp ./timewarp.sqlite3 ./blobs debug <run_id>
```

Resume deterministically and run a what‑if injection (using this demo’s factory):

```
timewarp ./timewarp.sqlite3 ./blobs resume <run_id> \
  --app examples.langgraph_demo.multi_agent_full:make_graph_multi \
  --thread t-demo --freeze-time

timewarp ./timewarp.sqlite3 ./blobs inject <run_id> <step> \
  --output alt.json \
  --app examples.langgraph_demo.multi_agent_full:make_graph_multi \
  --thread t-demo --record-fork --freeze-time
```

Prompt overrides (DSPy-style)
-----------------------------

Provide a JSON mapping of agent/node name to an override spec. A spec can be a string (treated as a system message or appended to a raw prompt) or an object with a mode and text.

Example overrides.json

```json
{
  "planner": { "mode": "prepend_system", "text": "Be concise and accurate." },
  "review": "Prefer bullet points"
}
``

Apply overrides during a non-recorded resume (tolerate prompt-hash diffs with --allow-diff):

```
timewarp ./timewarp.sqlite3 ./blobs resume <run_id> \
  --app examples.langgraph_demo.multi_agent_full:make_graph_multi \
  --thread t-demo \
  --prompt-overrides ./overrides.json \
  --allow-diff
```

Fork and record a branch with overrides using inject:

```
timewarp ./timewarp.sqlite3 ./blobs inject <run_id> 0 \
  --prompt-overrides ./overrides.json \
  --app examples.langgraph_demo.multi_agent_full:make_graph_multi \
  --thread t-demo \
  --allow-diff \
  --record-fork
```

Or use the dedicated helper:

```
timewarp ./timewarp.sqlite3 ./blobs dspy fork <run_id> \
  --app examples.langgraph_demo.multi_agent_full:make_graph_multi \
  --overrides ./overrides.json \
  --thread t-demo \
  --allow-diff \
  --record-fork
```

Each forked run is labeled with `branch_of=<baseline>` and `override_step=prompt_overrides`, so you can diff the branch against the baseline:

```
timewarp ./timewarp.sqlite3 ./blobs diff <baseline_run_id> <fork_run_id> --json
```

Event Filters Cheatsheet
------------------------

Focus on specific slices of the run quickly:

```
# Only TOOL events (MCP) from the tooling node
timewarp ./timewarp.sqlite3 ./blobs events <run_id> \
  --type TOOL --tool-kind MCP --node tooling --json

# Only LLM events from the planner node
timewarp ./timewarp.sqlite3 ./blobs events <run_id> --type LLM --node planner --json

# HITL interrupts from the human node
timewarp ./timewarp.sqlite3 ./blobs events <run_id> --type HITL --node human --json

# LLM events emitted inside the review subgraph (match by namespace)
timewarp ./timewarp.sqlite3 ./blobs events <run_id> --type LLM --namespace review --json

# All DECISION anchors, useful to understand routing and joins
timewarp ./timewarp.sqlite3 ./blobs events <run_id> --type DECISION --json
```

Notes
-----

- The demo records one sync run (no async nodes) to keep `.invoke()` compatible,
  then tries an async run with `make_graph_multi(include_async=True)` using `.ainvoke()`.
- If your environment does not provide `graph.astream`, the async run is skipped.
- When using `wrap(...)` without an explicit `LocalStore`, the default DB path is
  `./timewarp.sqlite3` (examples above use that). Earlier examples may reference
  `./timewarp.db`; both are supported if you pass matching paths on the CLI.

MCP Example (optional)
----------------------

When `langgraph` and `langchain-mcp-adapters` are available, you can run the MCP demo app:

```
# Record a run using the MCP example app
python - <<'PY'
from pathlib import Path
from timewarp.store import LocalStore
from timewarp.events import Run
from timewarp.langgraph import LangGraphRecorder
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
from timewarp.bindings import bind_langgraph_playback

# Define an installer with the standard 3‑arg signature
def installer(llm, tool, memory) -> None:
    bind_langgraph_playback(my_graph, llm, tool, memory)

session = replayer.resume(
    run_id, from_step=None, thread_id="t-1", install_wrappers=installer, freeze_time=True
)
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

CLI Internals (Contributors)
----------------------------

- Entry point: `timewarp.cli:main` dispatches to a decomposed CLI under `timewarp/cli/`.
- Commands: `timewarp/cli/commands/*` implement subcommands (list, events, tools, diff, resume, inject, export, fsck, debug).
- Helpers: `timewarp/cli/helpers/*` contains small utilities used by the CLI only:
  - `jsonio`: `print_json`, `dumps_text`, `loads_file` (orjson‑backed).
  - `state`: `format_state_pretty`, `dump_event_output_to_file`.
  - `events`: `filter_events`.
  - `filters`: `parse_list_filters`.
- Stability: these helper modules are implementation details for the CLI and are not part of the public API; they may change between versions.
- Programmatic use: prefer the core modules and top‑level exports (`timewarp.events`, `timewarp.store`, `timewarp.diff`, `timewarp.replay`, `timewarp.langgraph`, etc.).
