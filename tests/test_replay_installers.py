from __future__ import annotations

from pathlib import Path

import pytest

from timewarp.bindings import bind_langgraph_playback
from timewarp.events import ActionType, BlobKind, Event, Run
from timewarp.replay import LLMPromptMismatch, PlaybackLLM, PlaybackTool, _EventCursor
from timewarp.store import LocalStore

try:
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    LC_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    LC_AVAILABLE = False


@pytest.mark.skipif(not LC_AVAILABLE, reason="langchain-core not installed")
def test_installers_patch_chatmodel_and_replay_output(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="demo")
    store.create_run(run)

    # Prepare a recorded LLM event at step 0 with a simple JSON payload
    payload = {"message": {"content": "recorded"}}
    bref = store.put_blob(run.run_id, 0, BlobKind.OUTPUT, __import__("orjson").dumps(payload))
    ev = Event(
        run_id=run.run_id,
        step=0,
        action_type=ActionType.LLM,
        actor="llm",
        output_ref=bref,
        hashes={"output": bref.sha256_hex},
        labels={},
    )
    store.append_event(ev)

    events = store.list_events(run.run_id)
    llm = PlaybackLLM(store=store, cursor=_EventCursor(events=events, action_type=ActionType.LLM))
    tool = PlaybackTool(
        store=store, cursor=_EventCursor(events=events, action_type=ActionType.TOOL)
    )

    teardown = bind_langgraph_playback(graph=None, llm=llm, tool=tool)
    try:
        model = FakeListChatModel(responses=["ignored"])  # model output should be ignored by patch
        out = model.invoke("hello")
        assert out == payload  # playback returned recorded JSON
    finally:
        teardown()


@pytest.mark.skipif(not LC_AVAILABLE, reason="langchain-core not installed")
def test_playbackllm_prompt_mismatch_raises(tmp_path: Path) -> None:
    store = LocalStore(db_path=tmp_path / "db.sqlite", blobs_root=tmp_path / "blobs")
    run = Run(project="p", name="demo")
    store.create_run(run)

    # Compute a prompt hash compatible with the PlaybackLLM hashing logic
    from timewarp.codec import to_bytes
    from timewarp.events import hash_bytes

    expected_prompt_hash = hash_bytes(to_bytes({"prompt": "expected"}))
    payload = {"message": {"content": "recorded"}}
    bref = store.put_blob(run.run_id, 0, BlobKind.OUTPUT, __import__("orjson").dumps(payload))
    ev = Event(
        run_id=run.run_id,
        step=0,
        action_type=ActionType.LLM,
        actor="llm",
        output_ref=bref,
        hashes={"output": bref.sha256_hex, "prompt": expected_prompt_hash},
        labels={},
    )
    store.append_event(ev)
    events = store.list_events(run.run_id)

    llm = PlaybackLLM(store=store, cursor=_EventCursor(events=events, action_type=ActionType.LLM))
    tool = PlaybackTool(
        store=store, cursor=_EventCursor(events=events, action_type=ActionType.TOOL)
    )
    teardown = bind_langgraph_playback(graph=None, llm=llm, tool=tool)
    try:
        model = FakeListChatModel(responses=["ignored"])  # patched path
        with pytest.raises(LLMPromptMismatch):
            _ = model.invoke("different")
    finally:
        teardown()
