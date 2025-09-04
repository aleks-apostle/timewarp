from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import UUID

from .codec import zstd_compress, zstd_decompress
from .events import BlobKind, BlobRef, Event, Run, hash_bytes
from .telemetry import record_event_span

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  project TEXT,
  name TEXT,
  framework TEXT,
  code_version TEXT,
  started_at TEXT,
  finished_at TEXT,
  status TEXT,
  labels TEXT,
  schema_version INTEGER
);

CREATE TABLE IF NOT EXISTS events (
  run_id TEXT NOT NULL,
  step INTEGER NOT NULL,
  action_type TEXT,
  actor TEXT,
  input_ref TEXT,
  output_ref TEXT,
  ts TEXT,
  rng_state BLOB,
  model_meta TEXT,
  hashes TEXT,
  parent_step INTEGER,
  labels TEXT,
  privacy_marks TEXT,
  schema_version INTEGER,
  tool_kind TEXT,
  tool_name TEXT,
  mcp_server TEXT,
  mcp_transport TEXT,
  tools_digest TEXT,
  mem_op TEXT,
  mem_scope TEXT,
  mem_space TEXT,
  mem_provider TEXT,
  query_id TEXT,
  retriever TEXT,
  top_k INTEGER,
  PRIMARY KEY (run_id, step)
);

CREATE INDEX IF NOT EXISTS events_by_actor ON events(run_id, actor);
CREATE INDEX IF NOT EXISTS events_by_type ON events(run_id, action_type);
CREATE INDEX IF NOT EXISTS runs_project_started ON runs(project, started_at);
-- JSON label helper indexes (best-effort; require SQLite JSON1)
CREATE INDEX IF NOT EXISTS idx_events_run_checkpoint ON events(
  run_id,
  json_extract(labels, '$.checkpoint_id')
);
CREATE INDEX IF NOT EXISTS idx_events_run_anchor ON events(
  run_id,
  json_extract(labels, '$.anchor_id')
);
"""


@dataclass
class LocalStore:
    """SQLite metadata + filesystem blobs.

    - `db_path`: path to SQLite file.
    - `blobs_root`: directory where blobs are stored (created if missing).
    """

    db_path: Path
    blobs_root: Path
    # Optional SQLite tuning knobs
    busy_timeout_ms: int | None = 5000

    def __post_init__(self) -> None:
        self.blobs_root.mkdir(parents=True, exist_ok=True)
        with self._conn() as con:
            con.executescript(_DDL)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        # Apply PRAGMAs on every connection (journal mode is persistent; sync is per-connection)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        # Apply busy timeout if configured (helps under concurrent writers)
        try:
            if isinstance(self.busy_timeout_ms, int) and self.busy_timeout_ms > 0:
                con.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)};")
        except Exception:
            pass
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def create_run(self, run: Run) -> None:
        with self._conn() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO runs(
                  run_id, project, name, framework, code_version, started_at,
                  finished_at, status, labels, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(run.run_id),
                    run.project,
                    run.name,
                    run.framework,
                    run.code_version,
                    run.started_at.isoformat(),
                    run.finished_at.isoformat() if run.finished_at else None,
                    run.status,
                    json.dumps(run.labels),
                    run.schema_version,
                ),
            )

    def list_runs(self, project: str | None = None) -> list[Run]:
        with self._conn() as con:
            if project is None:
                cur = con.execute(
                    "SELECT run_id, project, name, framework, code_version, started_at, "
                    "finished_at, status, labels, schema_version FROM runs "
                    "ORDER BY started_at DESC"
                )
            else:
                cur = con.execute(
                    "SELECT run_id, project, name, framework, code_version, started_at, "
                    "finished_at, status, labels, schema_version FROM runs "
                    "WHERE project = ? ORDER BY started_at DESC",
                    (project,),
                )
            rows = cur.fetchall()
        runs: list[Run] = []
        for (
            run_id,
            project,
            name,
            framework,
            code_version,
            started_at,
            finished_at,
            status,
            labels,
            schema_version,
        ) in rows:
            runs.append(
                Run(
                    run_id=UUID(run_id),
                    project=project,
                    name=name,
                    framework=framework,
                    code_version=code_version,
                    started_at=datetime_from_iso(started_at),
                    finished_at=datetime_from_iso(finished_at) if finished_at else None,
                    status=status,
                    labels=json.loads(labels) if labels else {},
                    schema_version=schema_version,
                )
            )
        return runs

    def append_event(self, ev: Event) -> None:
        # Emit an optional OTel span and persist trace/span ids into model_meta if available
        ev_to_store = ev
        with record_event_span(ev) as ids:
            trace_id_hex, span_id_hex = ids
            if trace_id_hex and span_id_hex:
                try:
                    meta = dict(ev.model_meta or {})
                    meta.setdefault("otel_trace_id", trace_id_hex)
                    meta.setdefault("otel_span_id", span_id_hex)
                    ev_to_store = ev.model_copy(update={"model_meta": meta})
                except Exception:
                    ev_to_store = ev
            # Ensure referenced blobs are finalized on disk before inserting event
            try:
                if ev_to_store.input_ref is not None:
                    self._finalize_blob_file(ev_to_store.input_ref)
                if ev_to_store.output_ref is not None:
                    self._finalize_blob_file(ev_to_store.output_ref)
            except Exception:
                # propagate after closing span context to avoid masking tracing
                raise
            with self._conn() as con:
                con.execute(
                    """
                INSERT INTO events (
                  run_id, step, action_type, actor, input_ref, output_ref, ts, rng_state,
                  model_meta, hashes, parent_step, labels, privacy_marks, schema_version,
                  tool_kind, tool_name, mcp_server, mcp_transport,
                  tools_digest, mem_op, mem_scope, mem_space,
                  mem_provider, query_id, retriever, top_k
                ) VALUES (
                  ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                    (
                        str(ev_to_store.run_id),
                        ev_to_store.step,
                        ev_to_store.action_type.value,
                        ev_to_store.actor,
                        ev_to_store.input_ref.model_dump_json() if ev_to_store.input_ref else None,
                        ev_to_store.output_ref.model_dump_json()
                        if ev_to_store.output_ref
                        else None,
                        ev_to_store.ts.isoformat(),
                        ev_to_store.rng_state,
                        json.dumps(ev_to_store.model_meta) if ev_to_store.model_meta else None,
                        json.dumps(ev_to_store.hashes),
                        ev_to_store.parent_step,
                        json.dumps(ev_to_store.labels),
                        json.dumps(ev_to_store.privacy_marks),
                        ev_to_store.schema_version,
                        ev_to_store.tool_kind,
                        ev_to_store.tool_name,
                        ev_to_store.mcp_server,
                        ev_to_store.mcp_transport,
                        ev_to_store.tools_digest,
                        ev_to_store.mem_op,
                        ev_to_store.mem_scope,
                        ev_to_store.mem_space,
                        ev_to_store.mem_provider,
                        ev_to_store.query_id,
                        ev_to_store.retriever,
                        ev_to_store.top_k,
                    ),
                )

    def append_events(self, events: list[Event]) -> None:
        """Append multiple events in a single transaction, preserving order.

        Emits OTel spans per event similarly to append_event.
        """
        if not events:
            return
        with self._conn() as con:
            for ev in events:
                ev_to_store = ev
                # Emit span and embed trace/span ids when available
                with record_event_span(ev) as ids:
                    trace_id_hex, span_id_hex = ids
                    if trace_id_hex and span_id_hex:
                        try:
                            meta = dict(ev.model_meta or {})
                            meta.setdefault("otel_trace_id", trace_id_hex)
                            meta.setdefault("otel_span_id", span_id_hex)
                            ev_to_store = ev.model_copy(update={"model_meta": meta})
                        except Exception:
                            ev_to_store = ev
                    # Ensure referenced blobs are finalized on disk before inserting event
                    if ev_to_store.input_ref is not None:
                        self._finalize_blob_file(ev_to_store.input_ref)
                    if ev_to_store.output_ref is not None:
                        self._finalize_blob_file(ev_to_store.output_ref)
                    con.execute(
                        """
                    INSERT INTO events (
                      run_id, step, action_type, actor, input_ref, output_ref, ts, rng_state,
                      model_meta, hashes, parent_step, labels, privacy_marks, schema_version,
                      tool_kind, tool_name, mcp_server, mcp_transport,
                      tools_digest, mem_op, mem_scope, mem_space,
                      mem_provider, query_id, retriever, top_k
                    ) VALUES (
                      ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                        (
                            str(ev_to_store.run_id),
                            ev_to_store.step,
                            ev_to_store.action_type.value,
                            ev_to_store.actor,
                            ev_to_store.input_ref.model_dump_json()
                            if ev_to_store.input_ref
                            else None,
                            ev_to_store.output_ref.model_dump_json()
                            if ev_to_store.output_ref
                            else None,
                            ev_to_store.ts.isoformat(),
                            ev_to_store.rng_state,
                            json.dumps(ev_to_store.model_meta) if ev_to_store.model_meta else None,
                            json.dumps(ev_to_store.hashes),
                            ev_to_store.parent_step,
                            json.dumps(ev_to_store.labels),
                            json.dumps(ev_to_store.privacy_marks),
                            ev_to_store.schema_version,
                            ev_to_store.tool_kind,
                            ev_to_store.tool_name,
                            ev_to_store.mcp_server,
                            ev_to_store.mcp_transport,
                            ev_to_store.tools_digest,
                            ev_to_store.mem_op,
                            ev_to_store.mem_scope,
                            ev_to_store.mem_space,
                            ev_to_store.mem_provider,
                            ev_to_store.query_id,
                            ev_to_store.retriever,
                            ev_to_store.top_k,
                        ),
                    )

    def list_events(self, run_id: UUID) -> list[Event]:
        with self._conn() as con:
            cur = con.execute(
                (
                    "SELECT run_id, step, action_type, actor, input_ref, output_ref, ts, "
                    "rng_state, model_meta, hashes, parent_step, labels, privacy_marks, "
                    "schema_version, tool_kind, tool_name, mcp_server, mcp_transport, "
                    "tools_digest, mem_op, mem_scope, mem_space, mem_provider, query_id, "
                    "retriever, top_k FROM events WHERE run_id=? ORDER BY step ASC"
                ),
                (str(run_id),),
            )
            rows = cur.fetchall()
        events: list[Event] = []
        for r in rows:
            (
                run_id_s,
                step,
                action_type,
                actor,
                input_ref,
                output_ref,
                ts,
                rng_state,
                model_meta,
                hashes,
                parent_step,
                labels,
                privacy_marks,
                schema_version,
                tool_kind,
                tool_name,
                mcp_server,
                mcp_transport,
                tools_digest,
                mem_op,
                mem_scope,
                mem_space,
                mem_provider,
                query_id,
                retriever,
                top_k,
            ) = r

            def parse_blob(s: str | None) -> BlobRef | None:
                if not s:
                    return None
                return BlobRef.model_validate_json(s)

            events.append(
                Event(
                    run_id=UUID(run_id_s),
                    step=step,
                    action_type=action_type,
                    actor=actor,
                    input_ref=parse_blob(input_ref),
                    output_ref=parse_blob(output_ref),
                    ts=datetime_from_iso(ts),
                    rng_state=rng_state,
                    model_meta=json.loads(model_meta) if model_meta else None,
                    hashes=json.loads(hashes) if hashes else {},
                    parent_step=parent_step,
                    labels=json.loads(labels) if labels else {},
                    privacy_marks=json.loads(privacy_marks) if privacy_marks else {},
                    schema_version=schema_version,
                    tool_kind=tool_kind,
                    tool_name=tool_name,
                    mcp_server=mcp_server,
                    mcp_transport=mcp_transport,
                    tools_digest=tools_digest,
                    mem_op=mem_op,
                    mem_scope=mem_scope,
                    mem_space=mem_space,
                    mem_provider=mem_provider,
                    query_id=query_id,
                    retriever=retriever,
                    top_k=top_k,
                )
            )
        return events

    def list_events_window(self, run_id: UUID, offset: int, limit: int) -> list[Event]:
        """Return a window of events ordered by step (offset/limit).

        Does not load the full run into memory. Useful for paging in CLIs.
        """
        off = max(0, int(offset))
        lim = max(1, int(limit))
        with self._conn() as con:
            cur = con.execute(
                (
                    "SELECT run_id, step, action_type, actor, input_ref, output_ref, ts, "
                    "rng_state, model_meta, hashes, parent_step, labels, privacy_marks, "
                    "schema_version, tool_kind, tool_name, mcp_server, mcp_transport, "
                    "tools_digest, mem_op, mem_scope, mem_space, mem_provider, query_id, "
                    "retriever, top_k FROM events WHERE run_id=? ORDER BY step ASC LIMIT ? OFFSET ?"
                ),
                (str(run_id), lim, off),
            )
            rows = cur.fetchall()
        events: list[Event] = []
        for r in rows:
            (
                run_id_s,
                step,
                action_type,
                actor,
                input_ref,
                output_ref,
                ts,
                rng_state,
                model_meta,
                hashes,
                parent_step,
                labels,
                privacy_marks,
                schema_version,
                tool_kind,
                tool_name,
                mcp_server,
                mcp_transport,
                tools_digest,
                mem_op,
                mem_scope,
                mem_space,
                mem_provider,
                query_id,
                retriever,
                top_k,
            ) = r

            def parse_blob(s: str | None) -> BlobRef | None:
                if not s:
                    return None
                return BlobRef.model_validate_json(s)

            events.append(
                Event(
                    run_id=UUID(run_id_s),
                    step=step,
                    action_type=action_type,
                    actor=actor,
                    input_ref=parse_blob(input_ref),
                    output_ref=parse_blob(output_ref),
                    ts=datetime_from_iso(ts),
                    rng_state=rng_state,
                    model_meta=json.loads(model_meta) if model_meta else None,
                    hashes=json.loads(hashes) if hashes else {},
                    parent_step=parent_step,
                    labels=json.loads(labels) if labels else {},
                    privacy_marks=json.loads(privacy_marks) if privacy_marks else {},
                    schema_version=schema_version,
                    tool_kind=tool_kind,
                    tool_name=tool_name,
                    mcp_server=mcp_server,
                    mcp_transport=mcp_transport,
                    tools_digest=tools_digest,
                    mem_op=mem_op,
                    mem_scope=mem_scope,
                    mem_space=mem_space,
                    mem_provider=mem_provider,
                    query_id=query_id,
                    retriever=retriever,
                    top_k=top_k,
                )
            )
        return events

    def count_events(self, run_id: UUID) -> int:
        with self._conn() as con:
            cur = con.execute("SELECT COUNT(*) FROM events WHERE run_id = ?", (str(run_id),))
            row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def last_event_ts(self, run_id: UUID) -> datetime | None:
        with self._conn() as con:
            cur = con.execute(
                "SELECT ts FROM events WHERE run_id = ? ORDER BY step DESC LIMIT 1",
                (str(run_id),),
            )
            row = cur.fetchone()
        if not row or row[0] is None:
            return None
        return datetime_from_iso(row[0])

    def put_blob(
        self,
        run_id: UUID,
        step: int,
        kind: BlobKind,
        payload: bytes,
        *,
        content_type: str | None = None,
        compress: bool = True,
    ) -> BlobRef:
        rel_dir = Path("runs") / str(run_id) / "events" / str(step)
        dir_path = self.blobs_root / rel_dir
        dir_path.mkdir(parents=True, exist_ok=True)
        filename = f"{kind.value}.bin"
        tmp_path = dir_path / (filename + ".tmp")
        data = zstd_compress(payload) if compress else payload
        # Write to a temp file first; finalize during event append
        tmp_path.write_bytes(data)
        return BlobRef(
            run_id=run_id,
            step=step,
            kind=kind,
            path=str(rel_dir / filename),
            size_bytes=len(data),
            content_type=content_type,
            compression="zstd" if compress else None,
            sha256_hex=hash_bytes(payload),
        )

    def get_blob(self, ref: BlobRef) -> bytes:
        # Attempt lazy finalization: if final file missing but tmp exists, rename now
        self._finalize_blob_file(ref)
        data = (self.blobs_root / ref.path).read_bytes()
        if ref.compression == "zstd":
            return zstd_decompress(data)
        return data

    # --- blob finalization helpers ---

    def _finalize_blob_file(self, ref: BlobRef) -> None:
        """Ensure the blob file exists at its final path; rename from .tmp if present.

        Raises FileNotFoundError if both final and temp files are missing.
        """
        final_path = self.blobs_root / ref.path
        if final_path.exists():
            return
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        # If a tmp file exists, atomically move into place
        if tmp_path.exists():
            tmp_path.replace(final_path)
            return
        # Nothing to finalize; report missing file
        raise FileNotFoundError(str(final_path))


def datetime_from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)
