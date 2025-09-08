Developer Notes

- CLI helpers (use these instead of duplicating logic):
  - `timewarp/cli/helpers/prompts.py`
    - `build_prompt_adapter(spec)` compiles a string or dict spec into a callable.
    - `load_prompt_overrides(path)` loads agent->spec JSON and compiles adapters.
    - Supported dict spec: `{mode: prepend_system|append_system|replace_system|append_prompt|replace_prompt, text: str}`.
  - `timewarp/cli/helpers/blobs.py`
    - `read_json_blob(store, ref)` reads and decodes JSON blobs via canonical codec.
  - `timewarp/cli/helpers/imports.py`
    - `load_factory("module:function")` imports a factory; raises clear errors on failures.

- Canonical JSON path
  - All JSON encoding/decoding flows through `timewarp.codec` (or `timewarp.utils.json` shim).
  - `utils.json.dumps/loads` are thin shims over `codec.to_bytes/from_bytes` to ensure orjson + sorted keys for deterministic hashing.

- Store insertion and observability
  - `LocalStore.append_event(s)` now delegates enrichment to `_prepare_event_for_insert`:
    - Embeds OTel trace/span ids into `model_meta` when available.
    - Finalizes referenced `.tmp` blobs before DB inserts.
  - `record_event_span(ev)` remains around inserts; spans are emitted best‑effort.

- Provenance consistency
  - `model_meta.timewarp_version` is stamped on messages, retrieval, and memory events.
  - `adapter_version` reflects the adapter/library version in use.

- Replay guarantees
  - Playback wrappers are required for runs containing LLM/TOOL events; CLIs bind these automatically.
  - Resume prefers the step‑0 SYS input blob when present; otherwise falls back to the first `input_ref`.

