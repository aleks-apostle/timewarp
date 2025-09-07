# Privacy Marks and Redaction

Timewarp helps you avoid leaking sensitive data by applying redaction at serialization time. You can declare `privacy_marks` that map dotted JSON paths to a redaction strategy.

Strategies
- `redact`: replace value with `[REDACTED]`
- `mask4`: for strings, keep last 4 characters (e.g., `***1234`); otherwise redact

Where redaction applies
- Recorder serializes inputs, outputs, and state snapshots through a redaction pass using the `privacy_marks` you supply. Redaction is best-effort and only affects persisted blobs; the in-memory objects in your app are unchanged.

Configuring privacy marks

Python (facade)
```python
from timewarp import wrap

rec = wrap(
    graph,
    project="billing",
    privacy_marks={
        "input.user.ssn": "mask4",
        "args.api_key": "redact",
        "message.content": "redact",       # redact message text in messages stream
        "values.customer.email": "mask4",   # redact in values snapshots
    },
    enable_record_taps=True,
)
_ = rec.invoke({"user": {"ssn": "000-11-1234"}}, config={"configurable": {"thread_id": "t-1"}})
```

Python (direct recorder)
```python
from timewarp.langgraph import LangGraphRecorder
rec = LangGraphRecorder(
    graph=compiled,
    store=store,
    run=run,
    privacy_marks={"input.user.ssn": "mask4", "args.api_key": "redact"},
    stream_modes=("updates", "messages", "values"),
)
```

Guidance
- Prefer redacting at the narrowest path that still meets policy (e.g., `input.payment.card` vs `input`).
- For tool calls, target `args.<field>` to handle argument blobs.
- For large histories, combine with `messages_pruner` to trim unneeded content.
- Do not include secrets in labels/metadata; labels are not redacted and used for indexing and filters.

Security model
- Redaction happens before bytes hit disk; hashes are computed on redacted content.
- Event timestamps and identifiers are never redacted.
- For multi-tenant/server deployments, pair redaction with store-level RBAC and retention policies.
