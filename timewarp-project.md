# Timewarp — Deterministic Replay & Time‑Travel Debugger for LLM Agent Workflows

> **Record every step. Rewind any step. Reproduce any run.**
> Timewarp adds event‑sourced logging and deterministic replay to agent frameworks (LangGraph first, LangChain optional), plus a CLI/TUI debugger for step‑through, diffs, and what‑if edits. It fills a well‑documented gap: mainstream tools visualize traces but don’t let you **replay them exactly**.;

---

## Table of contents

* [Why Timewarp](#why-timewarp)
* [Features](#features)
* [Architecture (how it works)](#architecture-how-it-works)
* [Project structure](#project-structure)
* [Tech stack](#tech-stack)
* [Install](#install)
* [Quickstart](#quickstart)
* [CLI / Debugger](#cli--debugger)
* [Data model](#data-model)
* [Determinism & concurrency model](#determinism--concurrency-model)
* [Storage backends](#storage-backends)
* [Adapters](#adapters)
* [Telemetry & exports](#telemetry--exports)
* [Security & privacy](#security--privacy)
* [Performance targets](#performance-targets)
* [Roadmap at a glance](#roadmap-at-a-glance)
* [Contributing & license](#contributing--license)

---

## Why Timewarp

Agent frameworks matured (LangGraph, LangChain, AutoGen, CrewAI), but **none** offers step‑perfect reproducibility with interactive time‑travel debugging. The gap matrix in the design doc shows “Deterministic Replay Debugging” as **absent across the board**—Timewarp is built to own that row.;

* **Problem:** flaky, non‑repro runs; tough root‑cause analysis; “works on my machine” for LLMs/tools.
* **Solution:** event‑sourced execution, deterministic replay, first‑divergence diffs, and a proper debugger. The **sequence diagram on page 24** and **component diagram on page 25** illustrate the core flow and separation of modules this repository implements.;

---

## Features

* **Event‑sourced recording** of LLM calls, tool calls, HITL steps, and state mutations with inputs, outputs, timestamps, hashes, and (when used) RNG/time/env captures.;
* **Deterministic replay** that injects recorded outputs (LLM/tools), restores seeded randomness, and lets you step/next/goto/continue, inspect state, and inject/skip events.;
* **Causal diff & delta debugging:** find the **first divergence** between two runs; binary‑search helpers to isolate minimal failing differences.;
* **HITL preservation & what‑ifs:** faithfully replay or override recorded human decisions.;
* **Observability:** OpenTelemetry spans per event; optional export to LangSmith.;

> Targets: ≤5–10% record overhead; fast replays; 1k+ event runs. See **build plan & perf targets** in the doc (Week‑by‑week plan; performance section).;

---

## Architecture (how it works)

**At a glance (from the design doc diagrams):**

* **Recording mode:** Agent → Timewarp Proxy → real tools/LLMs. Proxy appends an **Event** for each step to the **Event Log** (store). *(Sequence diagram, p.24.)*;
* **Replay mode:** The **Replay Engine** loads events, reconstructs state up to a breakpoint, then runs forward with **recorded outputs** (no live side‑effects), enabling time‑travel debugging. *(Sequence & component diagrams, pp.24–25.)*;

**Core modules:** `Adapters` (LangGraph first), `Determinism Layer`, `Event Log`, `Replay Engine`, `CLI/TUI Debugger`, `Telemetry & Exports`. *(Components detailed across pp.24–31.)*;

---

## Project structure

> Mirrors the reference skeleton in the design doc (pp.33–34), extended to optional server/UI while keeping the core library separable.;

```
timewarp/
├─ README.md
├─ LICENSE
├─ CONTRIBUTING.md
├─ pyproject.toml
├─ packages/
│  ├─ timewarp-core/        # Python SDK: adapters, determinism, event log, replay, diff, telemetry
│  │  ├─ timewarp/
│  │  │  ├─ logging.py      # Event / Run models, storage drivers (SQLite dev; Postgres driver optional)
│  │  │  ├─ replay.py       # ReplayEngine APIs (step/next/goto/diff/inject/skip)
│  │  │  ├─ adapters/
│  │  │  │  ├─ langgraph.py # Node hooks (record/replay) and HITL capture
│  │  │  │  └─ langchain.py # Callback handler + Playback LLM / tool wrappers
│  │  │  ├─ telemetry.py    # OpenTelemetry export + LangSmith export
│  │  │  ├─ cli.py          # Debugger REPL (list/show/goto/step/state/diff/inject)
│  │  │  └─ __init__.py
│  │  └─ pyproject.toml
│  ├─ timewarp-cli/         # Thin console entrypoint (can be folded into core)
│  ├─ timewarp-server/      # (Optional) API + headless replay workers + storage (Postgres/S3)
│  └─ timewarp-ui/          # (Optional) Web UI for timelines, diffs, HITL
├─ examples/                # Demo agents and recorded runs
├─ tests/                   # Unit, integration, property-based, performance tests
└─ docs/                    # MkDocs/Sphinx site (Quickstart, API, Ops)
```

---

## Tech stack

* **Language:** Python 3.11/3.12 (typed).;
* **Core libs:** `pydantic v2` (schemas), `orjson` (fast JSON), `zstandard` (compression), `deepdiff`/`difflib` (diff), `rich`/`prompt_toolkit` (TUI). *(Packaging & skeleton sections.)*;
* **Adapters:** `langgraph` (primary), `langchain` (optional).;
* **Storage:** Dev = SQLite (WAL). Prod option = PostgreSQL (+ S3 blobs). *(Storage section pp.29–31.)*;
* **Telemetry:** `opentelemetry-sdk` + OTLP exporter; optional LangSmith export.;
* **Server (optional):** FastAPI/gRPC, SQLAlchemy, `psycopg`, `boto3`. *(Server/UI envisioned as add‑on; commercialization notes p.35.)*;

---

## Install

```bash
# Core SDK & CLI
pip install timewarp-agents
# Optional adapters
pip install langgraph langchain
# Optional telemetry
pip install opentelemetry-sdk opentelemetry-exporter-otlp
```

> Packaging and CI targets (Py3.11/3.12) are outlined in the doc’s packaging section.;

---

## Quickstart

```python
from timewarp import TimewarpProxy, Replay
from my_graph import graph  # e.g., a LangGraph

# Record a run
agent = TimewarpProxy.wrap(graph, project="billing", labels={"git":"abc123"})
result = agent.invoke({"input": "Find anomalies in August invoices"})
run_id = agent.last_run_id

# Debug / replay later
dbg = Replay.load(run_id)
dbg.goto(12).inspect_state()
dbg.step()                 # advance one event
dbg.diff("other-run-id")   # first divergence vs. another run
```

> The API shape follows the **Quickstart & API sketches** in the design doc (pp.31–34), and the **sequence diagram on p.24** explains why replay can inject recorded outputs deterministically.;

---

## CLI / Debugger

```bash
timewarp list --project billing
timewarp debug <run_id>
# inside REPL:
list          # show timeline of steps
show 17       # show a specific event
goto 12       # jump to step 12
step          # execute next event
state         # inspect current agent state
diff <runB>   # side-by-side and first-divergence
inject 20 ... # inject an alternate output at step 20
```

> Commands align with the **interactive CLI plan** (pp.31–34)—step/next/goto/diff/inject—and with HITL pause points preserved from frameworks.;

---

## Data model

**Event** *(immutable, append‑only)*
Fields include: `run_id`, `step` (canonical order), `action_type` (LLM/TOOL/DECISION/HITL/SNAPSHOT), `actor`, `input_ref`, `output_ref`, `ts`, optional `rng_state`, `model_meta`, `hashes`, `labels`, `schema_version`. Large payloads are stored as blobs and referenced. *(See model outlines and storage notes pp.25–31.)*;

**Run**
Metadata (`framework`, `code_version`, `labels`), `started_at`, `finished_at`, `status`. *(Run tracking & schemas pp.29–31.)*;

---

## Determinism & concurrency model

* **LLMs & tools:** record **verbatim outputs** and inject them during replay; log model/prompt hashes; seed RNG/time/env. *(Determinism plan pp.26–27.)*;
* **Concurrency:** if the live run had parallel branches, the **recorded completion order** defines the canonical timeline; replay serializes to that order to reproduce final state. *(Execution model p.26.)*;
* **Error model:** clear `ReplayError` taxonomy (missing blob, version mismatch, adapter mismatch) with actionable messages. *(Replay error handling p.27.)*;

---

## Storage backends

* **Local/dev (default):** SQLite (WAL), JSON(‑like) payloads (optionally compressed). *(Schema + SQLite rationale pp.29–31.)*;
* **Production option:** PostgreSQL for metadata + S3‑compatible blobs; optional search index for prompts/outputs. *(Productionization beyond v0.1, consistent with storage design.)*;

---

## Adapters

* **LangGraph (primary):** node start/finish hooks, HITL capture, state snapshots (optional optimization), bypass external calls on replay. *(LangGraph fit & adapter plan pp.24–29.)*;
* **LangChain:** callback handler for record; **PlaybackLLM** and **PlaybackTool** wrappers for replay. *(Adapter idea p.29.)*;

---

## Telemetry & exports

* **OpenTelemetry:** spans per event with attributes (`run_id`, `step`, `actor`, `action_type`, `replay=true/false`), plus **span links** that correlate original and replay traces. *(Observability section pp.27–28.)*;
* **LangSmith export:** optional exporter for teams using LangSmith UIs. *(pp.27–28.)*;

---

## Security & privacy

* **Redaction & encryption:** mark sensitive fields (`privacy_marks`) for masking or encryption at source; never log fields configured as secret. *(Policy & safety notes p.28.)*;
* **Replay sandboxing:** replays run **without external egress** and skip live side‑effects, using recorded outputs to preserve determinism. *(Replay guidance pp.10–12, 26–27.)*;

---

## Performance targets

* **Recording overhead:** ≤5–10% for \~1k events (batching, WAL, compression).
* **Replay speed:** near real‑time (no network); optional snapshots to accelerate long `goto`s.
* **Scale:** multi‑GB traces, ≥1k events/run.
  Targets and benchmark plan are documented in the performance section and week‑by‑week build plan.;

---

## Roadmap at a glance

* **v0.1 (core):** Event Log, deterministic replay, CLI stepping/diff, LangGraph adapter, OTel export, tests/docs. *(4‑week plan & skeleton pp.2, 31–36.)*;
* **Team mode (optional):** server API (Postgres+S3), headless replay workers, web UI for timelines/diffs/HITL, RBAC/redaction/retention. *(Commercialization & UI notes p.35; policy p.28.)*;

---

## Contributing & license

* **Contributing:** Standard PR/issue workflow; tests & linters must pass (see `CONTRIBUTING.md`). *(Packaging/tests/docs guidance pp.31–36.)*;
* **License:** OSS‑friendly (MIT/Apache‑2.0). See `LICENSE`.

---

### References (from the project design doc)

* **Sequence diagram (record vs. replay):** *page 24* — shapes the Timewarp Proxy ↔ Event Log ↔ Replay Engine flow implemented here.;
* **Component diagram:** *page 25* — clarifies module boundaries (Proxy, Event Log, Replay, Debug CLI, OTel).;
* **Gap matrix:** *pages 6–7* — establishes the absence of deterministic replay in current frameworks.;
* **Storage, determinism, tests & packaging:** *pages 26–31, 31–36* — inform schema, replay rules, benchmarks, and repo skeleton.;

---

> **Status**: Actively evolving. The core aligns 1:1 with the design doc’s v0.1 scope and extends cleanly toward a multi‑user server/UI when needed—without compromising the determinism guarantees at the heart of Timewarp.;
