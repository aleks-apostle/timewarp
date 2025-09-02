from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

from examples.langgraph_demo.app import make_graph
from timewarp.adapters.installers import bind_langgraph_playback
from timewarp.replay import LangGraphReplayer
from timewarp.store import LocalStore


def main(run_id_str: str, thread_id: str | None = None, from_step: int | None = None) -> None:
    db = Path("./timewarp.db")
    blobs = Path("./blobs")
    store = LocalStore(db_path=db, blobs_root=blobs)
    graph = make_graph()
    replayer = LangGraphReplayer(graph=graph, store=store)

    def installer(llm, tool) -> None:
        bind_langgraph_playback(graph, llm, tool)

    session = replayer.resume(UUID(run_id_str), from_step, thread_id, install_wrappers=installer)
    print("Resumed run:", run_id_str)
    print("checkpoint_id=", session.checkpoint_id)
    print("result:", session.result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m examples.resume_demo <run_id> [thread_id] [from_step]")
        raise SystemExit(2)
    rid = sys.argv[1]
    tid = sys.argv[2] if len(sys.argv) > 2 else None
    step = int(sys.argv[3]) if len(sys.argv) > 3 else None
    main(rid, tid, step)
