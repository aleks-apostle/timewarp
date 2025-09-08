from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from .jsonio import loads_file


def build_prompt_adapter(item: object) -> Callable[[Any], Any]:
    """Compile a prompt override spec into a callable adapter.

    Supported specs:
    - str: for list messages, prepend a system message; for str prompts, append text.
    - dict: {"mode": one of {prepend_system, append_system, replace_system,
              append_prompt, replace_prompt}, "text": str}

    Behavior matches existing per-CLI implementations to avoid regressions.
    """

    def _identity(x: Any) -> Any:
        return x

    # Simple string: prepend system for messages; append for string prompts
    if isinstance(item, str):
        text = item

        def _adapter(x: Any) -> Any:
            if isinstance(x, list):
                return [{"role": "system", "content": text}, *list(x)]
            if isinstance(x, str):
                return str(x) + "\n\n" + text
            return x

        return _adapter

    # Object spec with explicit mode and text
    if isinstance(item, dict):
        mode = str(item.get("mode", "prepend_system")).lower()
        text = str(item.get("text", ""))

        def _adapter(x: Any) -> Any:
            if isinstance(x, list):
                msgs = list(x)
                if mode == "append_system":
                    msgs = [*msgs, {"role": "system", "content": text}]
                elif mode == "replace_system":
                    replaced = False
                    out = []
                    for m in msgs:
                        if not replaced and isinstance(m, dict) and m.get("role") == "system":
                            out.append({"role": "system", "content": text})
                            replaced = True
                        else:
                            out.append(m)
                    msgs = out if replaced else ([{"role": "system", "content": text}, *out])
                else:  # prepend_system (default)
                    msgs = [{"role": "system", "content": text}, *msgs]
                return msgs
            if isinstance(x, str):
                if mode in {"append_prompt", "append_system"}:
                    return str(x) + "\n\n" + text
                elif mode == "replace_prompt":
                    return text
                else:  # prepend_system / prepend_prompt
                    return text + "\n\n" + str(x)
            return x

        return _adapter

    return _identity


def load_prompt_overrides(path: Path) -> dict[str, Callable[[Any], Any]]:
    """Load agent->prompt override specs from JSON and compile adapters.

    Returns a mapping of agent name to a callable that transforms messages/prompt.
    Raises on invalid file contents; callers may handle exceptions for CLI UX.
    """
    obj = loads_file(Path(path))
    if not isinstance(obj, dict):
        raise ValueError("Invalid prompt overrides: expected object mapping agent -> spec")
    out: dict[str, Callable[[Any], Any]] = {}
    for k, v in obj.items():
        out[str(k)] = build_prompt_adapter(v)
    return out
