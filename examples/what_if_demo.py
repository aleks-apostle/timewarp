from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

import orjson as _orjson

from examples.langgraph_demo.app import make_graph
from timewarp.bindings import bind_langgraph_playback
from timewarp.replay import LangGraphReplayer
from timewarp.store import LocalStore


def main(run_id_str: str, step: int, replacement_json: str, thread_id: str | None = None) -> None:
    db = Path("./timewarp.db")
    blobs = Path("./blobs")
    store = LocalStore(db_path=db, blobs_root=blobs)
    graph = make_graph()
    replayer = LangGraphReplayer(graph=graph, store=store)

    def installer(llm, tool) -> None:
        bind_langgraph_playback(graph, llm, tool)

    replacement = _orjson.loads(replacement_json)
    new_id = replayer.fork_with_injection(
        UUID(run_id_str),
        at_step=step,
        replacement=replacement,
        thread_id=thread_id,
        install_wrappers=installer,
    )
    print("Prepared fork run:", new_id)
    print("Note: record the forked execution by running your app with the recorder.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        msg = (
            "Usage: python -m examples.what_if_demo "
            "<run_id> <step> '<replacement_json>' [thread_id]"
        )
        print(msg)
        raise SystemExit(2)
    rid = sys.argv[1]
    stp = int(sys.argv[2])
    repl = sys.argv[3]
    tid = sys.argv[4] if len(sys.argv) > 4 else None
    main(rid, stp, repl, tid)
