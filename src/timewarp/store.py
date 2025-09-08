from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final
from uuid import UUID

from .codec import zstd_compress, zstd_decompress
from .events import BlobKind, BlobRef, Event, Run, hash_bytes
from .telemetry import record_event_span
from .utils.logging import log_warn_once

# Event table column order used by INSERTs/SELECTs.
_EVENT_COLS: Final[str] = (
    "run_id, step, action_type, actor, input_ref, output_ref, ts, rng_state, "
    "model_meta, hashes, parent_step, labels, privacy_marks, schema_version, "
    "tool_kind, tool_name, mcp_server, mcp_transport, tools_digest, mem_op, "
    "mem_scope, mem_space, mem_provider, query_id, retriever, top_k"
)

_NUM_EVENT_COLS: Final[int] = _EVENT_COLS.count(",") + 1


def _event_to_db_tuple(ev: Event) -> tuple[object, ...]:
    """Map Event to a DB tuple matching the INSERT column order."""
    return (
        str(ev.run_id),
        ev.step,
        ev.action_type.value,
        ev.actor,
        ev.input_ref.model_dump_json() if ev.input_ref else None,
        ev.output_ref.model_dump_json() if ev.output_ref else None,
        ev.ts.isoformat(),
        ev.rng_state,
        json.dumps(ev.model_meta) if ev.model_meta else None,
        json.dumps(ev.hashes),
        ev.parent_step,
        json.dumps(ev.labels),
        json.dumps(ev.privacy_marks),
        ev.schema_version,
        ev.tool_kind,
        ev.tool_name,
        ev.mcp_server,
        ev.mcp_transport,
        ev.tools_digest,
        ev.mem_op,
        ev.mem_scope,
        ev.mem_space,
        ev.mem_provider,
        ev.query_id,
        ev.retriever,
        ev.top_k,
    )


def _row_to_event(row: tuple[Any, ...]) -> Event:
    """Map a SELECT row (ordered by _EVENT_COLS) to an Event."""
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
    ) = row

    def parse_blob(s: str | None) -> BlobRef | None:
        if not s:
            return None
        return BlobRef.model_validate_json(s)

    return Event(
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
            # Apply non-destructive migrations before verifying schema
            self._migrate_if_needed(con)
            # Best-effort: create JSON1-based indexes when available
            self._create_json_indexes_if_available(con)
            # Finally, verify schema has required columns after migrations
            self._verify_schema(con)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path)
        # Apply PRAGMAs on every connection (journal mode is persistent; sync is per-connection)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            log_warn_once("sqlite.pragmas.journal_sync", None)
        # Apply busy timeout if configured (helps under concurrent writers)
        try:
            if isinstance(self.busy_timeout_ms, int) and self.busy_timeout_ms > 0:
                con.execute(f"PRAGMA busy_timeout={int(self.busy_timeout_ms)};")
        except Exception:
            log_warn_once("sqlite.pragmas.busy_timeout", None, {"ms": self.busy_timeout_ms})
        # Additional recommended PRAGMAs (best-effort)
        try:
            con.execute("PRAGMA foreign_keys=ON;")
        except Exception as e:
            log_warn_once("sqlite.pragmas.foreign_keys", e)
        try:
            con.execute("PRAGMA temp_store=MEMORY;")
        except Exception as e:
            log_warn_once("sqlite.pragmas.temp_store", e)
        try:
            # Limit WAL/journal growth on long sessions (value in bytes)
            con.execute("PRAGMA journal_size_limit=67108864;")  # 64 MiB
        except Exception as e:
            log_warn_once("sqlite.pragmas.journal_size_limit", e)
        try:
            yield con
            con.commit()
        except Exception:
            try:
                con.rollback()
            except Exception as e2:
                log_warn_once("sqlite.conn.rollback_failed", e2)
            raise
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
        # Emit an optional OTel span, enrich model_meta, then insert; finalize blobs after insert.
        with record_event_span(ev) as ids:
            ev_to_store = self._prepare_event_for_insert(ev, ids)
            with self._conn() as con:
                # Monotonic step guard per run
                try:
                    cur = con.execute(
                        "SELECT MAX(step) FROM events WHERE run_id=?", (str(ev_to_store.run_id),)
                    )
                    row = cur.fetchone()
                    max_step = int(row[0]) if row and row[0] is not None else -1
                    if int(ev_to_store.step) <= max_step:
                        raise RuntimeError(
                            f"events out-of-order or duplicate step for run {ev_to_store.run_id}: "
                            f"db_max={max_step} got={ev_to_store.step}"
                        )
                except Exception as e:
                    if isinstance(e, RuntimeError):
                        raise
                    # best-effort; on failure, proceed and let PK constraint enforce
                    log_warn_once("sqlite.monotonic.guard.single", e)
                try:
                    con.execute(
                        (
                            f"INSERT INTO events ({_EVENT_COLS}) VALUES ("
                            + ",".join(["?"] * _NUM_EVENT_COLS)
                            + ")"
                        ),
                        _event_to_db_tuple(ev_to_store),
                    )
                except sqlite3.IntegrityError as e:
                    # Re-raise a clearer domain error for PK conflicts
                    raise RuntimeError(
                        f"duplicate or out-of-order step for run {ev_to_store.run_id}: "
                        f"step={ev_to_store.step}"
                    ) from e
                # Finalize blobs only after a successful insert
                # Let exceptions roll back the transaction
                self._finalize_if_tmp(ev_to_store.input_ref)
                self._finalize_if_tmp(ev_to_store.output_ref)

    def append_events(self, events: list[Event]) -> None:
        """Append multiple events in a single transaction, preserving order.

        Emits OTel spans per event similarly to append_event.
        """
        if not events:
            return
        with self._conn() as con:
            # Pre-check monotonic order per run_id
            try:
                # Group events by run_id preserving input order
                by_run: dict[str, list[Event]] = {}
                for ev in events:
                    by_run.setdefault(str(ev.run_id), []).append(ev)
                for run_id_s, evs in by_run.items():
                    # Ensure ascending strictly within batch as given
                    last = -1
                    for ev in evs:
                        s = int(ev.step)
                        if s <= last:
                            raise RuntimeError(
                                f"batch steps not strictly increasing for run {run_id_s}: "
                                f"prev={last} got={s}"
                            )
                        last = s
                    # Compare first batch step against existing max in DB
                    cur = con.execute("SELECT MAX(step) FROM events WHERE run_id=?", (run_id_s,))
                    row = cur.fetchone()
                    max_step = int(row[0]) if row and row[0] is not None else -1
                    if int(evs[0].step) <= max_step:
                        raise RuntimeError(
                            f"events out-of-order or duplicate step for run {run_id_s}: "
                            f"db_max={max_step} first_batch_step={int(evs[0].step)}"
                        )
            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise
                log_warn_once("sqlite.monotonic.guard.batch", e)
            for ev in events:
                # Emit span, enrich meta, then insert; finalize blobs after insert
                with record_event_span(ev) as ids:
                    ev_to_store = self._prepare_event_for_insert(ev, ids)
                    try:
                        con.execute(
                            (
                                f"INSERT INTO events ({_EVENT_COLS}) VALUES ("
                                + ",".join(["?"] * _NUM_EVENT_COLS)
                                + ")"
                            ),
                            _event_to_db_tuple(ev_to_store),
                        )
                    except sqlite3.IntegrityError as e:
                        raise RuntimeError(
                            f"duplicate or out-of-order step for run {ev_to_store.run_id}: "
                            f"step={ev_to_store.step}"
                        ) from e
                    # Finalize any referenced blobs post-insert
                    self._finalize_if_tmp(ev_to_store.input_ref)
                    self._finalize_if_tmp(ev_to_store.output_ref)

    def list_events(self, run_id: UUID) -> list[Event]:
        with self._conn() as con:
            cur = con.execute(
                (f"SELECT {_EVENT_COLS} FROM events WHERE run_id=? ORDER BY step ASC"),
                (str(run_id),),
            )
            rows = cur.fetchall()
        return [_row_to_event(r) for r in rows]

    def list_events_window(self, run_id: UUID, offset: int, limit: int) -> list[Event]:
        """Return a window of events ordered by step (offset/limit).

        Does not load the full run into memory. Useful for paging in CLIs.
        """
        off = max(0, int(offset))
        lim = max(1, int(limit))
        with self._conn() as con:
            cur = con.execute(
                (
                    f"SELECT {_EVENT_COLS} FROM events WHERE run_id=? "
                    "ORDER BY step ASC LIMIT ? OFFSET ?"
                ),
                (str(run_id), lim, off),
            )
            rows = cur.fetchall()
        return [_row_to_event(r) for r in rows]

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

    def _verify_schema(self, con: sqlite3.Connection) -> None:
        """Ensure required columns exist on core tables.

        If required columns are missing after migrations, raise with a clear message.
        """
        try:
            cur = con.execute("PRAGMA table_info(events)")
            cols = {str(r[1]) for r in cur.fetchall()}
        except Exception:
            # Table will be created by _DDL above; nothing to verify yet
            return
        required = {
            "tools_digest",
            "mem_op",
            "mem_scope",
            "mem_space",
            "mem_provider",
            "query_id",
            "retriever",
            "top_k",
        }
        missing = sorted(list(required - cols))
        if missing:
            raise RuntimeError(
                "Timewarp DB schema is outdated. Missing columns: "
                + ", ".join(missing)
                + ". Please re-initialize your DB or upgrade via migrations."
            )

    def _migrate_if_needed(self, con: sqlite3.Connection) -> None:
        """Apply non-destructive migrations and set PRAGMA user_version.

        Adds missing columns on events and updates user_version to the current schema.
        """
        try:
            # Determine current columns
            cur = con.execute("PRAGMA table_info(events)")
            cols = {str(r[1]) for r in cur.fetchall()}
        except Exception as e:
            log_warn_once("sqlite.migrate.table_info", e)
            cols = set()

        # Planned columns and their SQL types
        needed: dict[str, str] = {
            "tools_digest": "TEXT",
            "mem_op": "TEXT",
            "mem_scope": "TEXT",
            "mem_space": "TEXT",
            "mem_provider": "TEXT",
            "query_id": "TEXT",
            "retriever": "TEXT",
            "top_k": "INTEGER",
        }
        for col, typ in needed.items():
            if col not in cols:
                try:
                    con.execute(f"ALTER TABLE events ADD COLUMN {col} {typ}")
                except Exception as e:
                    # If the column already exists or ALTER fails, warn once and continue
                    log_warn_once("sqlite.migrate.add_column." + col, e)

        # Set/upgrade user_version (best-effort)
        try:
            target = 3
            ver_row = con.execute("PRAGMA user_version").fetchone()
            current = int(ver_row[0]) if ver_row and ver_row[0] is not None else 0
            if current < target:
                con.execute(f"PRAGMA user_version={target}")
        except Exception as e:
            log_warn_once("sqlite.migrate.user_version", e)

    def _create_json_indexes_if_available(self, con: sqlite3.Connection) -> None:
        """Create JSON1-dependent indexes when the JSON1 extension is available.

        Best-effort: if JSON1 is not available, skip and warn once.
        """
        try:
            # Probe json_extract
            con.execute("SELECT json_extract('[]', '$[0]')")
        except Exception as e:
            log_warn_once("sqlite.json1.unavailable", e)
            return
        try:
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_run_checkpoint ON events(
                  run_id,
                  json_extract(labels, '$.checkpoint_id')
                );
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_run_anchor ON events(
                  run_id,
                  json_extract(labels, '$.anchor_id')
                );
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_events_run_thread_step ON events(
                  run_id,
                  json_extract(labels, '$.thread_id'),
                  step
                );
                """
            )
        except Exception as e:
            log_warn_once("sqlite.json1.index_create", e)

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

    def _finalize_if_tmp(self, ref: BlobRef | None) -> None:
        """Finalize referenced blob if present.

        No-op when ref is None. Uses the store's blob root for resolution.
        """
        if ref is None:
            return
        self._finalize_blob_file(ref)

    # --- insert preparation helpers ---

    def _prepare_event_for_insert(
        self, ev: Event, ids: tuple[str | None, str | None] | None = None
    ) -> Event:
        """Prepare an Event for insertion by enriching meta and finalizing blobs.

        - Optionally embeds OTel trace/span ids from `ids` into `model_meta` if present.
        - Finalizes any `.tmp` blob files referenced by input/output.
        - Returns an updated Event; does not mutate the input instance.
        """
        ev_to_store = ev
        # Enrich model_meta with otel ids when provided
        try:
            if ids is not None:
                trace_id_hex, span_id_hex = ids
                if trace_id_hex and span_id_hex:
                    meta = dict(ev.model_meta or {})
                    meta.setdefault("otel_trace_id", trace_id_hex)
                    meta.setdefault("otel_span_id", span_id_hex)
                    ev_to_store = ev.model_copy(update={"model_meta": meta})
        except Exception:
            ev_to_store = ev
        # Do not finalize blobs here; defer until after a successful insert
        return ev_to_store


def datetime_from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)
