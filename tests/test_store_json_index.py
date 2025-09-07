from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from timewarp.store import LocalStore


def test_store_json_thread_index(tmp_path: Path) -> None:
    db = tmp_path / "tw.sqlite3"
    blobs = tmp_path / "blobs"
    # Create store (invokes index creation best-effort)
    LocalStore(db_path=db, blobs_root=blobs)
    con = sqlite3.connect(db)
    # Detect JSON1 support
    has_json1 = True
    try:
        con.execute("SELECT json_extract('{\"x\":1}', '$.x')")
    except Exception:
        has_json1 = False
    if not has_json1:
        pytest.skip("sqlite JSON1 extension not available; skipping index assertion")
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
        ("idx_events_run_thread_step",),
    )
    row = cur.fetchone()
    assert row is not None and row[0] == "idx_events_run_thread_step"
