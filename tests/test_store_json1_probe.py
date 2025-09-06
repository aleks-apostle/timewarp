from __future__ import annotations

from pathlib import Path
from typing import Any

from timewarp.store import LocalStore


def test_json1_probe_best_effort(tmp_path: Path, monkeypatch: Any) -> None:
    db_path = tmp_path / "tw.sqlite3"
    blobs_root = tmp_path / "blobs"

    # Should not raise even if JSON1 is unavailable per probe
    store = LocalStore(db_path=db_path, blobs_root=blobs_root)

    class DummyCon:
        def execute(self, sql: str, *args: Any, **kwargs: Any):  # type: ignore[override]
            if sql.strip().startswith("SELECT json_extract("):
                raise RuntimeError("json1 missing")
            # ignore other statements
            return self

        def fetchone(self):  # pragma: no cover - not used here
            return None

    # Call helper directly with a connection that lacks JSON1
    store._create_json_indexes_if_available(DummyCon())
