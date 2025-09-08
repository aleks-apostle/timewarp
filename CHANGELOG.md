## Unreleased

## 0.2.0 - 2025-09-08

Enhancements
- Replay safety: add no-network guard (`timewarp.replay.no_network.no_network`), and `LangGraphReplayer.resume(..., no_network=True)` with CLI flag `--no-network`.
- Storage safety: finalize blob files only after successful DB inserts; rollback on failures.
- Error clarity: clearer duplicate/out-of-order step errors with run_id/step details.
- Hashing coherence: standardize prompt/tools/prompt_ctx hashing via utils; preserve aggregated hash semantics for streamed messages.
- Privacy: normalize_bytes remains the single redaction gate; add tests for nested paths and list indices.
- Diagnostics: one-time warnings in wrappers for non-fatal validation/override issues.
- CLI UX: default DB (`timewarp.sqlite3`) and blobs (`./blobs`) paths; flags still override.
- Telemetry: one-line info when OTel tracing is enabled (provider + sampling), no secrets.
- Typing polish: mark core constants as Final; keep mypy --strict green.

Internal
- Add unit tests for no-network guard and redaction path handling.
