from __future__ import annotations

from uuid import UUID

from ..events import ActionType, BlobKind


class ReplayError(Exception):
    """Base class for replay-related errors."""


class MissingBlob(ReplayError):
    def __init__(self, *, run_id: UUID, step: int, kind: BlobKind, path: str) -> None:
        self.run_id = run_id
        self.step = step
        self.kind = kind
        self.path = path
        super().__init__(f"Missing blob for run={run_id} step={step} kind={kind.value} path={path}")


class SchemaMismatch(ReplayError):
    def __init__(self, message: str, *, step: int | None = None) -> None:
        self.step = step
        super().__init__(message)


class AdapterInvariant(ReplayError):
    def __init__(self, message: str, *, step: int | None = None) -> None:
        self.step = step
        super().__init__(message)


class MissingRecordedEvent(ReplayError):
    def __init__(self, *, run_id: UUID, after_step: int, action_type: ActionType) -> None:
        self.run_id = run_id
        self.after_step = after_step
        self.action_type = action_type
        super().__init__(
            f"Missing recorded {action_type.value} event after step={after_step} in run={run_id}"
        )


class LLMPromptMismatch(ReplayError):
    def __init__(self, step: int, *, expected_hash: str | None, got_hash: str | None) -> None:
        self.step = step
        self.expected_hash = expected_hash
        self.got_hash = got_hash
        super().__init__(
            f"LLM prompt mismatch at step={step}: expected={expected_hash} got={got_hash}"
        )


class ToolArgsMismatch(ReplayError):
    def __init__(self, step: int, *, expected_hash: str | None, got_hash: str | None) -> None:
        self.step = step
        self.expected_hash = expected_hash
        self.got_hash = got_hash
        super().__init__(
            f"TOOL args mismatch at step={step}: expected={expected_hash} got={got_hash}"
        )


class ToolsDigestMismatch(ReplayError):
    def __init__(self, step: int, *, expected_digest: str | None, got_digest: str | None) -> None:
        self.step = step
        self.expected_digest = expected_digest
        self.got_digest = got_digest
        super().__init__(
            f"LLM tools digest mismatch at step={step}: expected={expected_digest} got={got_digest}"
        )


class PromptContextMismatch(ReplayError):
    def __init__(self, step: int, *, expected_hash: str | None, got_hash: str | None) -> None:
        self.step = step
        self.expected_hash = expected_hash
        self.got_hash = got_hash
        super().__init__(
            f"LLM prompt_ctx mismatch at step={step}: expected={expected_hash} got={got_hash}"
        )


class RetrievalQueryMismatch(ReplayError):
    def __init__(self, step: int, *, expected_hash: str | None, got_hash: str | None) -> None:
        self.step = step
        self.expected_hash = expected_hash
        self.got_hash = got_hash
        super().__init__(
            f"RETRIEVAL query mismatch at step={step}: expected={expected_hash} got={got_hash}"
        )


class RetrievalPolicyMismatch(ReplayError):
    def __init__(
        self, step: int, *, field: str, expected: str | int | None, got: str | int | None
    ) -> None:
        self.step = step
        self.field = field
        self.expected = expected
        self.got = got
        super().__init__(
            f"RETRIEVAL policy mismatch at step={step} ({field}): expected={expected} got={got}"
        )


class ModelMetaMismatch(ReplayError):
    def __init__(self, step: int, *, diffs: list[str]) -> None:
        self.step = step
        self.diffs = diffs
        msg = ", ".join(diffs) if diffs else "model_meta mismatch"
        super().__init__(f"Model meta mismatch at step={step}: {msg}")
