from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from .codec import to_bytes
from .determinism import now as tw_now


class ActionType(StrEnum):
    LLM = "LLM"
    TOOL = "TOOL"
    DECISION = "DECISION"
    HITL = "HITL"
    SNAPSHOT = "SNAPSHOT"
    SYS = "SYS"
    ERROR = "ERROR"
    MEMORY = "MEMORY"
    RETRIEVAL = "RETRIEVAL"


class BlobKind(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    STATE = "state"
    MEMORY = "memory"


class BlobRef(BaseModel):
    """Reference to a blob stored in the blob store.

    path is relative to the configured blob root.
    """

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    step: int
    kind: BlobKind
    path: str
    size_bytes: int
    content_type: str | None = None
    compression: Literal["zstd"] | None = None
    sha256_hex: str


class Run(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: UUID = Field(default_factory=uuid4)
    project: str | None = None
    name: str | None = None
    framework: str | None = None
    code_version: str | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    status: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    schema_version: int = 3


class Event(BaseModel):
    """Canonical event record.

    Large payloads are referenced via BlobRef and not embedded.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: UUID
    step: int
    action_type: ActionType
    actor: str
    input_ref: BlobRef | None = None
    output_ref: BlobRef | None = None
    ts: datetime = Field(default_factory=tw_now)
    rng_state: bytes | None = None
    model_meta: dict[str, Any] | None = None
    hashes: dict[str, str] = Field(default_factory=dict)
    parent_step: int | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    privacy_marks: dict[str, str] = Field(default_factory=dict)
    schema_version: int = 3

    # Adapter-specific observational metadata (kept explicit to avoid nesting arbitrary blobs)
    tool_kind: str | None = None  # e.g., "MCP" for MCP tool calls
    tool_name: str | None = None
    mcp_server: str | None = None  # URL or server id when tool_kind=="MCP"
    mcp_transport: str | None = None  # stdio | streamable_http
    # Prompt/tools context digest (stable hash of tools list/specs)
    tools_digest: str | None = None
    # Memory / retrieval specific observational fields
    mem_op: str | None = None  # PUT | UPDATE | DELETE | EVICT | READ
    mem_scope: str | None = None  # short | working | long
    mem_space: str | None = None  # logical namespace/container
    mem_provider: str | None = None  # LangGraphState | Mem0 | LlamaIndex | Custom
    query_id: str | None = None  # retrieval request id
    retriever: str | None = None  # vector | hybrid | router
    top_k: int | None = None

    _canonical_bytes: bytes | None = PrivateAttr(default=None)

    def canonical_bytes(self) -> bytes:
        """Stable JSON bytes of the event excluding private attrs.

        Uses pydantic's JSON-mode dump to normalize datetimes, UUIDs, etc.
        Then sorts keys to ensure deterministic hashing.
        """

        if self._canonical_bytes is None:
            obj = self.model_dump(mode="json")
            self._canonical_bytes = to_bytes(obj)
        return self._canonical_bytes

    def sha256_hex(self) -> str:
        return sha256(self.canonical_bytes()).hexdigest()


def hash_bytes(data: bytes) -> str:
    return sha256(data).hexdigest()


def redact(obj: Any, privacy_marks: dict[str, str]) -> Any:
    """Apply redaction to a JSON-serializable object.

    privacy_marks maps dotted paths (e.g., "args.ssn") to strategies:
    - "redact": replace value with "[REDACTED]"
    - "mask4": keep last 4 characters for strings, else redact
    Unknown strategies fall back to redact.
    """

    if not privacy_marks:
        return obj

    def apply(path: str, value: Any, strategy: str) -> Any:
        if strategy == "mask4" and isinstance(value, str) and len(value) > 4:
            return f"***{value[-4:]}"
        # default redact
        return "[REDACTED]"

    def walk(current: Any, base: str) -> Any:
        if isinstance(current, dict):
            out: dict[str, Any] = {}
            for k, v in current.items():
                p = f"{base}.{k}" if base else k
                if p in privacy_marks:
                    out[k] = apply(p, v, privacy_marks[p])
                else:
                    out[k] = walk(v, p)
            return out
        if isinstance(current, list):
            return [walk(v, f"{base}[{i}]") for i, v in enumerate(current)]
        return current

    return walk(obj, "")
