from __future__ import annotations

from pathlib import Path

import orjson as _orjson

from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.replay import ModelMetaMismatch, PlaybackLLM, _EventCursor
from timewarp.store import LocalStore


def _mk_recorded_llm_event(
    store: LocalStore, run: Run, step: int, meta: dict[str, object]
) -> Event:
    b = store.put_blob(run.run_id, step, BlobKind.OUTPUT, _orjson.dumps({"text": "ok"}))
    ev = Event(
        run_id=run.run_id,
        step=step,
        action_type=ActionType.LLM,
        actor="model",
        output_ref=b,
        hashes={"output": b.sha256_hex},
        model_meta=meta,
    )
    store.append_event(ev)
    return ev


def test_playback_llm_strict_meta_passes_on_match(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="r")
    store.create_run(run)
    # Single LLM event with recorded meta
    _ = _mk_recorded_llm_event(
        store,
        run,
        step=0,
        meta={"provider": "openai", "model": "gpt-4o", "temperature": 0.2},
    )
    events = store.list_events(run.run_id)
    cur = _EventCursor(events=events, action_type=ActionType.LLM)
    llm = PlaybackLLM(store=store, cursor=cur, strict_meta=True)
    out = llm.invoke(
        "prompt", _tw_model_meta={"provider": "openai", "model": "gpt-4o", "temperature": 0.2}
    )
    assert isinstance(out, dict)
    assert out.get("text") == "ok"


def test_playback_llm_strict_meta_raises_on_mismatch(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="r")
    store.create_run(run)
    _ = _mk_recorded_llm_event(
        store,
        run,
        step=0,
        meta={"provider": "openai", "model": "gpt-4o", "top_p": 0.9},
    )
    events = store.list_events(run.run_id)
    cur = _EventCursor(events=events, action_type=ActionType.LLM)
    llm = PlaybackLLM(store=store, cursor=cur, strict_meta=True)
    try:
        _ = llm.invoke("prompt", _tw_model_meta={"provider": "anthropic", "model": "claude"})
    except ModelMetaMismatch as e:
        # Helpful diff mentions keys
        msg = str(e)
        assert "provider" in msg or "model" in msg
    else:
        raise AssertionError("expected ModelMetaMismatch")
