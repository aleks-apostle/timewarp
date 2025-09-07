from __future__ import annotations

from timewarp import telemetry


def test_telemetry_sampling_alias_on() -> None:
    # Should not raise even when otel SDK is missing; best-effort behavior
    telemetry.configure(sampling="on")
    telemetry.configure(sampling="ratio(0.05)")
