from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from uuid import UUID

from ...store import LocalStore
from ..helpers.jsonio import print_json


def _handler(args: argparse.Namespace, store: LocalStore) -> int:
    run_id = UUID(args.run_id)
    events = store.list_events(run_id)
    # Gather referenced blob relative paths
    referenced: set[str] = set()
    for e in events:
        if e.input_ref:
            referenced.add(e.input_ref.path)
        if e.output_ref:
            referenced.add(e.output_ref.path)
    # Verify and optionally repair
    missing: list[str] = []
    repaired: list[str] = []
    for rel in sorted(referenced):
        final_path = Path(args.blobs) / rel
        if final_path.exists():
            continue
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        if tmp_path.exists() and bool(getattr(args, "repair", False)):
            try:
                tmp_path.replace(final_path)
                repaired.append(str(rel))
                continue
            except Exception:
                pass
        missing.append(str(rel))
    # Optionally gc-orphans: files on disk not referenced
    orphans: list[str] = []
    if bool(getattr(args, "gc_orphans", False)):
        try:
            all_files: list[Path] = []
            root = Path(args.blobs)
            for path_item in root.rglob("*.bin"):
                all_files.append(path_item)
            for path_item in root.rglob("*.bin.tmp"):
                all_files.append(path_item)
            ref_abs = {str(Path(args.blobs) / r) for r in referenced}
            # Exclude repaired final files
            ref_abs.update({str(Path(args.blobs) / r) for r in repaired})
            for path_item in all_files:
                if str(path_item) not in ref_abs:
                    # Apply grace period to avoid racing with in-flight writes
                    try:
                        grace = float(getattr(args, "grace", 5))
                    except Exception:
                        grace = 5.0
                    try:
                        mtime = path_item.stat().st_mtime
                    except Exception:
                        mtime = 0.0
                    if time.time() - mtime < grace:
                        # Defer deletion within grace window
                        continue
                    orphans.append(str(path_item.relative_to(root)))
                    try:
                        os.remove(path_item)
                    except Exception:
                        pass
        except Exception:
            pass
    print_json({"missing": missing, "repaired": repaired, "orphans_gc": orphans})
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    fsp = sub.add_parser(
        "fsck", help="Verify blobs referenced by a run and optionally repair/gc orphans"
    )
    fsp.add_argument("run_id", help="Run ID to check")
    fsp.add_argument(
        "--repair",
        dest="repair",
        action="store_true",
        help="Repair missing final blobs using .tmp files when available",
    )
    fsp.add_argument(
        "--gc-orphans",
        dest="gc_orphans",
        action="store_true",
        help="Garbage-collect orphaned blobs not referenced by the run",
    )
    fsp.add_argument(
        "--grace",
        dest="grace",
        type=float,
        default=5.0,
        help="Grace period in seconds; skip deleting orphans newer than this",
    )
    fsp.set_defaults(func=_handler)
