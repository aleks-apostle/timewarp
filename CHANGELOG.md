## Unreleased

- LangGraph memory events now stamp `timewarp_version` in `model_meta` for provenance parity with messages/retrieval.
- LocalStore insertion DRY refactor: added `_prepare_event_for_insert` to embed OTel ids and finalize blobs; reused by `append_event` and `append_events`.
- CLI `events` output shows compact labels (`sm=`, `ns=`, `thr=`) for improved scanability; `--json` unchanged.
- Replay input selection prefers step-0 SYS input blob when present, improving robustness for runs with multiple input-like blobs.

