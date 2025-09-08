OpenTelemetry Quickstart
========================

Timewarp emits spans per event (record and replay). Replay spans use Span Links
to connect to the original recorded span when the trace/span IDs were present
on the recorded event.

Install optional extras (one-time):

```
uv pip install -e .[otel]
```

Minimal configuration in your app:

```
from timewarp.telemetry import configure

# Console exporter (stdout) example
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

configure(exporter=ConsoleSpanExporter())

# Now run your recorder or replay; spans will be emitted per event
```

OTLP exporter example (to a collector):

```
from timewarp.telemetry import configure
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

configure(exporter=OTLPSpanExporter(endpoint="http://localhost:4318/v1/traces"))
```

Attributes follow the `tw.*` namespace and include:
- `tw.run_id`, `tw.step`, `tw.action_type`, `tw.actor`, `tw.replay`
- `tw.namespace`, `tw.thread_id`, `tw.checkpoint_id`, `tw.anchor_id`, `tw.branch_of`
- `tw.hash.output`, `tw.hash.state`, `tw.hash.prompt`

Notes
- Telemetry is best-effort; missing SDKs or exporter errors do not break recording.
- Span Links are attached only when recorded spans carried OTel IDs in `model_meta`.
 - Store insert refactor: `LocalStore.append_event(s)` still creates spans and embeds `trace_id`/`span_id` into `model_meta` via a shared helper.
