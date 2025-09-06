from __future__ import annotations

import sqlite3
from pathlib import Path

from timewarp.store import LocalStore


def _create_legacy_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
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
              -- intentionally missing: tools_digest, mem_op, mem_scope, mem_space, mem_provider,
              -- query_id, retriever, top_k
              PRIMARY KEY (run_id, step)
            );
            """
        )
        con.commit()
    finally:
        con.close()


def test_store_migrates_missing_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "tw.sqlite3"
    blobs_root = tmp_path / "blobs"
    _create_legacy_db(db_path)

    # Construct LocalStore; should perform migrations without raising
    LocalStore(db_path=db_path, blobs_root=blobs_root)
    # Verify columns exist now
    with sqlite3.connect(db_path) as con:
        cols = {row[1] for row in con.execute("PRAGMA table_info(events)").fetchall()}
    for col in (
        "tools_digest",
        "mem_op",
        "mem_scope",
        "mem_space",
        "mem_provider",
        "query_id",
        "retriever",
        "top_k",
    ):
        assert col in cols

    # user_version should be set to at least 3
    with sqlite3.connect(db_path) as con:
        ver = con.execute("PRAGMA user_version").fetchone()[0]
        assert int(ver) >= 3
