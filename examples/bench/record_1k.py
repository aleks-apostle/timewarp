from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from timewarp.events import Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _GenGraph:
    """Simple generator-based graph that yields N values updates."""

    def __init__(self, n: int) -> None:
        self.n = n

    def stream(self, inputs: dict[str, Any], config: dict[str, Any], **_: Any) -> Iterable[Any]:
        for i in range(self.n):
            yield ("values", {"values": {"i": i}})

    def get_state(self, config: dict[str, Any]) -> dict[str, Any]:
        return {"values": {"done": True}}


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--db", default="./timewarp.db")
    ap.add_argument("--blobs", default="./blobs")
    args = ap.parse_args()

    store = LocalStore(db_path=Path(args.db), blobs_root=Path(args.blobs))
    run = Run(project="bench", name=f"n={args.n},batch={args.batch}", framework="langgraph")
    rec = LangGraphRecorder(
        graph=_GenGraph(args.n),
        store=store,
        run=run,
        stream_modes=("values",),
        snapshot_every=1000000,
        event_batch_size=args.batch,
    )
    t0 = time.perf_counter()
    _ = rec.invoke({"start": True}, config={})
    dt = time.perf_counter() - t0
    print(f"Recorded {args.n} events in {dt:.3f}s  batch={args.batch}")


if __name__ == "__main__":
    main()
