from __future__ import annotations

from pathlib import Path
from typing import Any

from timewarp.bindings import bind_langgraph_playback
from timewarp.codec import to_bytes
from timewarp.events import ActionType, BlobKind, Event, Run, hash_bytes
from timewarp.replay.exceptions import LLMPromptMismatch
from timewarp.replay.wrappers import PlaybackLLM, _EventCursor
from timewarp.store import LocalStore


def _mk_store(tmp_path: Path) -> LocalStore:
    return LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")


def _mk_llm_event(
    store: LocalStore, run: Run, step: int, *, prompt_hash: str, labels: dict[str, str]
) -> Event:
    out_blob = store.put_blob(run.run_id, step, BlobKind.OUTPUT, to_bytes({"ok": True}))
    return Event(
        run_id=run.run_id,
        step=step,
        action_type=ActionType.LLM,
        actor=labels.get("node", "llm"),
        output_ref=out_blob,
        hashes={"output": out_blob.sha256_hex, "prompt": prompt_hash},
        labels=labels,
    )


def test_prompt_override_applied_and_validated(tmp_path: Path) -> None:
    store = _mk_store(tmp_path)
    run = Run(project="p", name="n")
    store.create_run(run)
    # Recorded prompt hash corresponds to {"prompt":"original"}
    rec_hash = hash_bytes(to_bytes({"prompt": "original"}))
    ev = _mk_llm_event(store, run, 1, prompt_hash=rec_hash, labels={"node": "agentX"})
    cur = _EventCursor(events=[ev], action_type=ActionType.LLM)

    # Override that changes the prompt (causing a mismatch)
    def adapter(x: Any) -> Any:
        if isinstance(x, str):
            return x + " MOD"
        return x

    # Strict (allow_diff=False) should raise mismatch
    llm_strict = PlaybackLLM(store=store, cursor=cur, prompt_overrides={"agentX": adapter})
    try:
        _ = llm_strict.invoke("original")
        raise AssertionError("expected LLMPromptMismatch")
    except LLMPromptMismatch:
        pass

    # With allow_diff=True, mismatch is tolerated and output is produced
    cur2 = _EventCursor(events=[ev], action_type=ActionType.LLM)
    llm_relaxed = PlaybackLLM(
        store=store, cursor=cur2, prompt_overrides={"agentX": adapter}, allow_diff=True
    )
    out = llm_relaxed.invoke("original")
    assert isinstance(out, dict) and out.get("ok") is True


def test_installers_threads_prompt_overrides(tmp_path: Path) -> None:
    store = _mk_store(tmp_path)
    run = Run(project="p", name="n")
    store.create_run(run)
    rec_hash = hash_bytes(to_bytes({"prompt": "p"}))
    ev = _mk_llm_event(store, run, 1, prompt_hash=rec_hash, labels={"node": "agentY"})
    cur = _EventCursor(events=[ev], action_type=ActionType.LLM)

    llm = PlaybackLLM(store=store, cursor=cur)
    tool = None  # type: ignore[assignment]
    # Bind with overrides; graph/memory unused in this test
    td = bind_langgraph_playback(
        graph=None,
        llm=llm,
        tool=tool,  # type: ignore[arg-type]
        memory=None,
        prompt_overrides={"agentY": lambda x: x},
    )
    try:
        assert "agentY" in llm.prompt_overrides
    finally:
        if callable(td):
            td()
