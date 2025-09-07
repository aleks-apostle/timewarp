from __future__ import annotations

from pathlib import Path
from typing import Any

from timewarp.events import ActionType, Run
from timewarp.langgraph import LangGraphRecorder
from timewarp.store import LocalStore


class _FakeUpdatesGraph:
    def __init__(self, updates: list[tuple[str, Any]]) -> None:
        self._updates = updates

    def stream(self, inputs: dict[str, Any], config: dict[str, Any] | None = None, **_: Any):
        yield from self._updates


def test_recorder_llm_params_in_model_meta(tmp_path: Path) -> None:
    # Craft an LLM-like update with metadata carrying params
    upd = (
        "updates",
        {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {
                "provider": "openai",
                "model": "gpt-4o",
                "temperature": 0.3,
                "top_p": 0.9,
                "tool_choice": "auto",
            },
        },
    )
    graph = _FakeUpdatesGraph([upd])
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="meta", framework="langgraph")
    rec = LangGraphRecorder(
        graph=graph,
        store=store,
        run=run,
        snapshot_every=0,
        stream_modes=("updates",),
        stream_subgraphs=False,
    )
    _ = rec.invoke({"x": 0}, config={})
    evs = store.list_events(run.run_id)
    # First is SYS input, second is our LLM event by heuristic
    llm_ev = next(e for e in evs if e.action_type is ActionType.LLM)
    assert llm_ev.model_meta is not None
    assert llm_ev.model_meta.get("provider") == "openai"
    assert llm_ev.model_meta.get("model") == "gpt-4o"
    # Params present
    assert llm_ev.model_meta.get("temperature") == 0.3
    assert llm_ev.model_meta.get("top_p") == 0.9
    assert llm_ev.model_meta.get("tool_choice") == "auto"
