# Timewarp Refactor Plan

This document is a comprehensive, phased blueprint to refactor the Timewarp codebase while preserving behavioral invariants: event‑sourced recording, deterministic replay with output injection, canonical completion order, and strong typing with Pydantic v2, orjson, and zstandard. Each phase contains scope, file map, concrete changes, compatibility notes, and acceptance criteria so you can execute it incrementally.

Status baseline (at time of writing)
- Python 3.11+, Pydantic v2, orjson, zstandard used correctly.
- Lint/type/tests: ruff passes, mypy --strict passes, pytest runs green (59 passed, 5 skipped).
- Large/complex modules: `src/timewarp/adapters/langgraph.py` (~2436 LOC), `src/timewarp/cli.py` (~1775 LOC), `src/timewarp/replay.py` (~899 LOC), `src/timewarp/adapters/installers.py` (~604 LOC).
- Observability: OpenTelemetry spans/links implemented best‑effort and optional via lazy imports.

Guiding constraints and invariants
- Style: Python 3.11/3.12, strict typing, prefer pure functions for core logic.
- Data models: Pydantic v2 (events/config), orjson for JSON, zstandard for blobs.
- Determinism: no hidden time/RNG/env reads in core logic; record RNG/time where required.
- Architecture: record (event‑sourcing) and replay (output injection), respect canonical recorded completion order during replay, snapshots may accelerate goto.
- Storage: SQLite/Postgres for metadata (SQLite in LocalStore), large payloads as blobs; support zstd.
- Workflows: always run `ruff format && ruff check --fix && mypy --strict && pytest -q` after each step. No network egress during replay.
- Diff UX: show first divergence and minimal failing delta.
- Security/Privacy: no secrets in logs, redaction via `privacy_marks` enforced where possible.
- Observability: OTel spans per event; span links from replay to original span.

What we’ll improve
1) Remove duplicated logic (hashing, DB mapping, stream processing, JSON printing).
2) Split overlong modules along logical boundaries.
3) Clarify and minimize backward‑compat pathways; add measured deprecations.
4) Keep tests passing at each phase; maintain deterministic behaviors.

Index of phases
- Phase 0: Baseline and guardrails
- Phase 1: Utilities extraction (JSON and hashing)
- Phase 2: Store API consolidation (Event↔DB mapping)
- Phase 3: Replay package split and cleanup
- Phase 4: LangGraph adapter modularization
- Phase 5: Installers split and dedupe
- Phase 6: CLI decomposition and small cleanups
- Phase 7: Anchors and diff consistency
- Phase 8: Deprecations and compatibility policy
- Phase 9: Observability hardening (polish)
- Phase 10: Documentation and tests

---

## Phase 0 — Baseline and guardrails

Goal
- Lock in reproducible workflows so every phase stays green.

Actions
- Adopt a recurring command: `ruff format && ruff check --fix && mypy --strict && pytest -q`.
- Ensure “replay” tasks use recorded outputs and do not egress network.

Acceptance criteria
- Commands run clean locally and in CI (if present) before and after each phase.

Notes
- The refactor will avoid changing external behavior; deprecations will be documented in Phase 8 and gated.

---

## Phase 1 — Utilities extraction (JSON and hashing)

Problem
- Repeated try/except orjson→json printing paths in CLI; custom hashing snippets reimplemented in multiple modules.

Changes
1) Add JSON helpers
   - New file: `src/timewarp/utils/json.py`
   - API:
     - `def dumps(obj: object) -> bytes:` canonical orjson dumps (sorted keys), fallback not used in core; CLI may catch exceptions when pretty printing.
     - `def loads(data: bytes) -> object:` wraps orjson.loads.
   - Replace ad‑hoc CLI JSON prints with this helper where appropriate.

2) Add hashing helpers
   - New file: `src/timewarp/utils/hashing.py`
   - Depends on `timewarp.codec.to_bytes` and `timewarp.events.hash_bytes`.
   - API:
     - `def hash_prompt(*, messages: object | None, prompt: object | None) -> str`
     - `def hash_tools_list(tools: object) -> str`
     - `def hash_prompt_ctx(*, messages: object, tools: object) -> str`
   - Migrate:
     - Replace `src/timewarp/replay.py` `_hash_prompt_like` and `_hash_tools_list` by these helpers.
     - Use these helpers in `src/timewarp/adapters/langgraph.py` for LLM/tool/prompts hashing and prompt_ctx construction.

3) Optional serialization helper (defer to Phase 4 if preferred)
   - New file: `src/timewarp/utils/serialize.py`
   - API:
     - `def normalize_bytes(obj: object, *, privacy_marks: dict[str, str], state_extractor: Callable[[object], dict[str, object] | None]) -> bytes`
   - Consolidate logic currently in `LangGraphRecorder._normalize_bytes`.

Files to touch
- Add: `src/timewarp/utils/json.py`, `src/timewarp/utils/hashing.py` (and optionally `src/timewarp/utils/serialize.py`).
- Update: `src/timewarp/replay.py:1`, `src/timewarp/adapters/langgraph.py:1`, `src/timewarp/cli.py:1` to import and use helpers.

Compatibility
- Pure internal refactor; public APIs unchanged.

Acceptance criteria
- All tests pass; no behavior changes.
- Grep for `_hash_prompt_like`, `_hash_tools_list`, and repeated orjson try/excepts in CLI shows usage removed or minimized.

---

## Phase 2 — Store API consolidation (Event↔DB mapping)

Problem
- Duplicate logic between `append_event` and `append_events` (finalizing blobs, tuple construction), and duplicate row→Event parsing between `list_events` and `list_events_window`.

Changes
1) Add internal helpers to `src/timewarp/store.py:1`
   - `_finalize_if_tmp(ref: BlobRef | None) -> None`
   - `_event_to_db_tuple(ev: Event) -> tuple[ ... ]` (exact order matching INSERT)
   - `_row_to_event(row: tuple) -> Event`

2) Refactor methods
   - `append_event` uses `_finalize_if_tmp` and `_event_to_db_tuple`.
   - `append_events` loops events using same helpers.
   - `list_events` and `list_events_window` map rows via `_row_to_event`.

3) Keep span recording logic intact (record_event_span) — only reduce duplication.

Files to touch
- Update: `src/timewarp/store.py:1` (introduce helpers and refactor call sites).

Compatibility
- Pure refactor; SQL schema and API stable.

Acceptance criteria
- Tests green; no functional changes.
- Single source of truth for Event↔DB mapping.

---

## Phase 3 — Replay package split and cleanup

Problem
- `src/timewarp/replay.py` contains exceptions, wrappers, LangGraph replayer, and navigation utilities; plus duplicated static resume function.

Changes
1) Split into submodules
   - New package: `src/timewarp/replay/`
     - `exceptions.py`: `ReplayError` and subclasses (`MissingBlob`, `SchemaMismatch`, `AdapterInvariant`, etc.).
     - `wrappers.py`: `_EventCursor`, `PlaybackLLM`, `PlaybackTool`, `PlaybackMemory`.
     - `langgraph.py`: `LangGraphReplayer` (`resume`, `fork_with_injection`).
     - `session.py`: `Replay` (navigation, overlays, `inspect_state`, `snapshot_now`).
     - `__init__.py`: re-export public symbols for backward compatibility.

2) Remove duplicate static `Replay.resume` vs `LangGraphReplayer.resume`
   - Keep one public entrypoint (`Replay.resume` calls into `LangGraphReplayer` under the hood), or expose a new top-level helper `resume_langgraph(...)` in `replay/__init__.py`.

3) Switch hashing calls to Phase 1 helpers.

Files to touch
- Move/update: `src/timewarp/replay.py:1` → new modules above.
- Update re-exports: `src/timewarp/__init__.py:1` to import from `timewarp.replay` package paths.

Compatibility
- Module-level imports remain stable via `__all__` in `src/timewarp/__init__.py` and `replay/__init__.py`.

Acceptance criteria
- Tests green; CLI resumes and injects unchanged.
- No import errors from previous public surfaces.

---

## Phase 4 — LangGraph adapter modularization

Problem
- `src/timewarp/adapters/langgraph.py` (~2.4k LOC) mixes stream ingestion (sync/async), messages aggregation, hashing, tool classification, memory synthesis, retrieval detection, anchor ID, serialization, batching.

Changes
1) Create a package: `src/timewarp/adapters/langgraph/`
   - `recorder.py`: `LangGraphRecorder` orchestration and public API.
   - `stream_sync.py`: sync stream (`.stream`) plumbing.
   - `stream_async.py`: async stream (`.astream`) plumbing.
   - `messages.py`: aggregation buffer and `_finalize_messages_aggregate`.
   - `hashing.py`: prompt/tools/prompt_ctx, source extraction; wraps Phase 1 util.
   - `memory.py`: synthesized memory events from state values.
   - `retrieval.py`: detection and emission of retrieval events.
   - `anchors.py`: `_make_anchor_id`; keep consistent with `diff.make_anchor_key`.
   - `serialize.py`: `_normalize_bytes`, `_extract_values`, `_extract_next_nodes`, `_extract_checkpoint_id`.
   - `classify.py`: `ToolClassifier` protocol and default classifier.
   - `batch.py`: batching buffer, `_append_event`, `_flush_events` unified for both sync/async.

2) Refactor `LangGraphRecorder` to delegate to these helpers
   - Sync and async paths share all normalization/extraction/hashing logic.
   - Use `store.append_events` to reduce write overhead while preserving order.

3) Ensure privacy marks redaction stays applied at serialization points.

Files to touch
- Move/update: `src/timewarp/adapters/langgraph.py:1` → modules above.
- Update imports in `src/timewarp/langgraph.py:1` (wrapper facade) if direct helpers referenced.

Compatibility
- Public class `LangGraphRecorder` stays in `timewarp.adapters.langgraph` via re-export.

Acceptance criteria
- Tests green, especially `tests/test_async_recorder.py` and taps tests.
- Sync/async runs record equivalent events, including labels, hashes, and message aggregates.

---

## Phase 5 — Installers split and dedupe

Problem
- `src/timewarp/adapters/installers.py` mixes: context/session staging, LangChain patching for record/playback, memory provider taps, and repeated patching code.

Changes
1) Create package: `src/timewarp/adapters/installers/`
   - `staging.py`: `_RecordingSession`, ContextVar session, legacy global deques, `begin_recording_session`, `stage_*`, `try_pop_*`.
   - `langchain_patch.py`: common patch helpers for BaseLanguageModel/BaseChatModel/BaseTool; generic `patch_method()` returning teardowns.
   - `record_taps.py`: `bind_langgraph_record()`; computes prompt/tool hashes using Phase 1 helpers; scopes to session when active else legacy fallback.
   - `playback.py`: `bind_langgraph_playback(graph, llm, tool, memory=None)`; patches LangChain models/tools; optional provider memory patches for playback.
   - `memory_patch.py`: best‑effort Mem0 and LlamaIndex tap/patch helpers used by `record_taps` and optionally by `playback` (read‑only in replay).
   - `__init__.py`: re-export current public functions for compatibility.

2) Reduce duplication
   - Use `patch_method()` in both record_taps and playback for LLM/Tool patching.
   - Share Mem0/LlamaIndex patchers between record and playback flavors (with correct behavior differences).

3) Back‑compat for global staging fallback
   - Keep legacy global deques; emit a warning when used without an active session.
   - Add env gate (e.g., `TIMEWARP_ALLOW_GLOBAL_TAPS=1` default on; planned off later per Phase 8).

Files to touch
- Move/update: `src/timewarp/adapters/installers.py:1` → modules above.
- Update imports: `src/timewarp/langgraph.py:1`, tests referencing installers, and `src/timewarp/replay.py:1` where installers are used.

Compatibility
- Preserve public symbol names via installers package `__init__.py`.

Acceptance criteria
- Tests referencing `bind_langgraph_record` and staging APIs pass.
- No regression in recorded prompt/tool hashes or provider taps.

---

## Phase 6 — CLI decomposition and small cleanups

Problem
- `src/timewarp/cli.py` packs all subcommands; repeated JSON printing and helpers; one unreachable branch for `fsck` (handler exists; parser does not).

Changes
1) Create CLI package
   - New package: `src/timewarp/cli/`
     - `main.py`: entrypoint with argparse; keeps `timewarp = "timewarp.cli:main"` by importing main from package.
     - `commands/`: `list_runs.py`, `events.py`, `tools.py`, `diff.py`, `resume.py`, `inject.py`, `export.py`, `debug.py`, optionally `fsck.py`.
     - `helpers/`: `jsonio.py` (wrapping `utils/json.py`), `events.py` (filters/printing), `tools.py` (tool extraction/summaries), `state.py` (pretty formatting), `io.py` (file dump helpers).

2) Decide `fsck`
   - Option A (recommended): add a real subcommand `fsck` with parser and wire it.
   - Option B: remove dead handler code.

3) Replace repeated orjson try/except print logic with `helpers/jsonio.py`.

Files to touch
- Move/update: `src/timewarp/cli.py:1` → package files above; keep `src/timewarp/cli.py` as a minimal shim calling `from .cli.main import main`.

Compatibility
- CLI flags/semantics remain stable; command names unchanged. If adding `fsck`, its flags will be documented.

Acceptance criteria
- CLI commands produce identical output (modulo whitespace) on representative runs.
- Dead `fsck` branch removed or a working `fsck` command added.

---

## Phase 7 — Anchors and diff consistency

Problem
- Anchor ID creation lives in adapter; diff builds anchor keys independently. Risk of drift.

Changes
- Create `src/timewarp/adapters/langgraph/anchors.py` exposing `_make_anchor_id`.
- Ensure `src/timewarp/diff.py:1` `make_anchor_key` remains aligned with adapter anchor scheme:
  - Prefer `labels["anchor_id"]` when present; fallback tuple `(action_type, actor, ns, tid, prompt_hash?)` unchanged.
- Add a small unit test to assert consistency for a synthetic event.

Files to touch
- Update: `src/timewarp/adapters/langgraph.py:1` (or new `anchors.py`), `src/timewarp/diff.py:1` comments to note alignment.

Acceptance criteria
- No change in diff outputs; ensures future adapter evolutions keep anchors consistent.

---

## Phase 8 — Deprecations and compatibility policy

Targets
- Installers `install_wrappers(llm, tool)` arity: standardize on 3‑arg `(llm, tool, memory)`.
- Global staging fallback: plan to disable by default in a later minor release.

Steps
1) Arity migration
   - Keep current dynamic arity in `LangGraphReplayer.fork_with_injection` with a warning when the 2‑arg variant is detected.
   - After 1–2 releases, remove the fallback and require 3‑arg signature.

2) Global staging fallback
   - When `_SESSION` is None and global deques are used, emit a warning suggesting `begin_recording_session()` or set `TIMEWARP_ALLOW_GLOBAL_TAPS=1`.
   - Document that default will flip to `0` in a future minor release.

Files to touch
- Update: `src/timewarp/replay.py:1` (warning on 2‑arg installers), `src/timewarp/adapters/installers/staging.py` (warning & env gate).
- Update docs (this file and README changelog section).

Acceptance criteria
- Tests unaffected (they already work with legacy behavior); warnings observable in logs when legacy pathways used.

---

## Phase 9 — Observability hardening (polish)

Problem
- Telemetry spans set attributes via inline try/except; repeated patterns.

Changes (non‑breaking, optional)
- Add tiny helpers in `src/timewarp/telemetry.py:1` to factor out attribute setting logic (already present as `_set_span_attrs`), keep as is or slightly dedupe callsites.
- Ensure span links on replay remain intact; validate with a smoke test if OTel is available.

Acceptance criteria
- No behavior regression; optional improvements only.

---

## Phase 10 — Documentation and tests

Changes
- Update `README.md:1` with new module paths and CLI structure where relevant.
- Add a `docs/CHANGELOG.md` entry documenting deprecations and migration steps.
- Consider brief module‑level docstrings for new packages (`replay`, `adapters/langgraph`, `adapters/installers`, `cli`).

Acceptance criteria
- All references accurate; development workflow documented.

---

## Detailed map of changes by file (summary)

New files/packages
- `src/timewarp/utils/json.py` — orjson helpers.
- `src/timewarp/utils/hashing.py` — prompt/tools/context hashing.
- `src/timewarp/utils/serialize.py` — normalize_bytes (optional).
- `src/timewarp/replay/__init__.py`, `exceptions.py`, `wrappers.py`, `langgraph.py`, `session.py` — replay split.
- `src/timewarp/adapters/langgraph/` — split recorder into `recorder.py`, `stream_sync.py`, `stream_async.py`, `messages.py`, `hashing.py`, `memory.py`, `retrieval.py`, `anchors.py`, `serialize.py`, `classify.py`, `batch.py`.
- `src/timewarp/adapters/installers/__init__.py`, `staging.py`, `langchain_patch.py`, `record_taps.py`, `playback.py`, `memory_patch.py`.
- `src/timewarp/cli/main.py`, `src/timewarp/cli/commands/*.py`, `src/timewarp/cli/helpers/*.py`.

Modified files (high‑level)
- `src/timewarp/replay.py:1` — replaced by package; keep shim re‑exports or delete once imports updated.
- `src/timewarp/adapters/langgraph.py:1` — replaced by package; keep shim re‑exports for BC.
- `src/timewarp/adapters/installers.py:1` — replaced by package; keep shim re‑exports for BC.
- `src/timewarp/cli.py:1` — replaced by package; keep shim calling `from .cli.main import main`.
- `src/timewarp/store.py:1` — add helpers to eliminate duplication.
- `src/timewarp/__init__.py:1` — update re‑exports to new modules.
- `src/timewarp/diff.py:1` — comments and minor alignment notes with anchors.
- `README.md:1` — doc adjustments.

Tests
- Re-run existing tests; only add unit tests for: hashing utils; optional anchors consistency.
- Avoid large test churn; legacy pathways remain until Phase 8 completion.

---

## Behavioral acceptance criteria per phase

All phases
- Determinism: replay uses recorded outputs; no network calls in replay paths.
- Ordering: recorded completion order is preserved in append and append_events.
- Redaction: `privacy_marks` applied on serialization in recorder paths.
- Observability: OTel span creation and replay span linking preserved.

Phase 1
- Replaced `_hash_prompt_like`/`_hash_tools_list` with `utils/hashing.py` helpers; no hash changes on existing runs.
- CLI JSON printing relies on `utils/json.py` (or keeps robust try/except in outer layers).

Phase 2
- One authoritative Event↔DB mapping code path; append/list functions delegate to helpers.
- Error/exception semantics unchanged.

Phase 3
- Imports such as `from timewarp.replay import Replay` and `Replay.resume(...)` keep working.
- `LangGraphReplayer` remains available from `timewarp.replay` (or `timewarp.replay.langgraph`).

Phase 4
- Sync/async recorders produce equivalent LLM/tool/MEMORY/RETRIEVAL/SYS events under same runtime.
- Messages aggregation identical (same chunks_count, prompt hashes, tools digest, prompt_ctx).

Phase 5
- `bind_langgraph_record`, `bind_langgraph_playback`, and staging APIs export from `timewarp.adapters.installers` unchanged.
- Legacy global staging deques still function with a warning when no session.

Phase 6
- CLI provides same commands and flags; if `fsck` enabled, it appears in help and works per existing handler logic.

Phase 7
- No change in diff outputs; anchor realignment operates as before.

Phase 8
- Warnings visible on legacy installer arity and global staging fallback.

Phase 9
- Optional; no behavior changes required.

Phase 10
- Documentation consistent with code; quickstarts work.

---

## Rollout and risk management

Strategy
- Incremental PRs per phase, keeping tests green.
- Avoid broad renames in a single change; introduce shims and re‑exports where needed.
- Add deprecation warnings before removing legacy code paths.

Primary risks and mitigations
- Subtle drift in hashing or aggregation: mitigate with unit tests around new helpers and verifying hashes on sample runs.
- Import breakage after splits: mitigate via re-exports in `__init__.py` and module shims.
- CLI behavior drift: validate outputs on representative runs; keep flag names identical.

---

## Execution checklist (repeat per phase)

1) Make scoped changes as per file map.
2) Run: `ruff format && ruff check --fix && mypy --strict && pytest -q`.
3) Manually test representative flows:
   - Record a small LangGraph run and resume it deterministically.
   - Use CLI: `list`, `events`, `tools`, `diff`, and if enabled `fsck`.
4) Confirm no network egress during replay (inspect code paths; rely on playback wrappers).
5) Review for `privacy_marks` propagation and redaction points.
6) Confirm OTel spans emit without hard dependency (optional in env).

This plan maintains deterministic behavior and the core user contracts while making the codebase smaller, clearer, and easier to extend.

