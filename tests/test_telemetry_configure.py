from __future__ import annotations

from uuid import uuid4

from timewarp.events import ActionType, Event
from timewarp.telemetry import configure, record_event_span


def test_telemetry_configure_noop_and_record_span() -> None:
    # Configure should be a no-op without OTel installed; must not raise
    provider = configure()
    # Provider may be None if otel not installed
    _ = provider
    # Create a minimal event and ensure record_event_span works
    ev = Event(run_id=uuid4(), step=0, action_type=ActionType.SYS, actor="graph")
    with record_event_span(ev) as ids:
        trace_id_hex, span_id_hex = ids
        # Accept either no-OTel (None,None) or OTel-present (hex strings)
        assert (trace_id_hex is None and span_id_hex is None) or (
            isinstance(trace_id_hex, str) and isinstance(span_id_hex, str)
        )
