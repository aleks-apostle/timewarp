from __future__ import annotations

from pathlib import Path

from examples.langgraph_demo.app import make_graph
from timewarp.adapters.langgraph import LangGraphRecorder
from timewarp.events import Run
from timewarp.store import LocalStore


def main() -> None:
    store = LocalStore(db_path=Path("./timewarp.db"), blobs_root=Path("./blobs"))
    run = Run(project="demo", name="record_demo", framework="langgraph")
    rec = LangGraphRecorder(
        graph=make_graph(),
        store=store,
        run=run,
        stream_modes=("updates", "values"),
        stream_subgraphs=True,
    )
    res = rec.invoke({"text": "hi"}, config={"configurable": {"thread_id": "t-1"}})
    print("Recorded run:", run.run_id)
    print("result:", res)


if __name__ == "__main__":
    main()
