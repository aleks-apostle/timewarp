from __future__ import annotations

from ..events import ActionType


def make_anchor_id(
    action_type: ActionType,
    actor: str,
    labels: dict[str, str],
    tool_name: str | None = None,
) -> str:
    """Construct a stable anchor id used for diff alignment.

    Keep in sync with `timewarp.diff.make_anchor_key` which prefers the
    explicit `labels["anchor_id"]` when comparing anchors.

    Format: "{thread_id}:{node}:{namespace}:{action}{:tool_name?}"
    """
    tid = labels.get("thread_id", "")
    node = labels.get("node", actor or "")
    ns = labels.get("namespace", "")
    tool = f":{tool_name}" if tool_name else ""
    return f"{tid}:{node}:{ns}:{action_type.value}{tool}"
