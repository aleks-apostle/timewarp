from __future__ import annotations

import importlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .utils.logging import log_warn_once

# Optional OpenTelemetry imports via importlib to keep mypy happy when not installed
_otel_trace: Any = None
_OtelLink: Any | None = None
_OtelSpanContext: Any | None = None
_OtelTraceFlags: Any | None = None
try:  # pragma: no cover - otel optional
    _otel_trace = importlib.import_module("opentelemetry.trace")
    _OtelLink = getattr(_otel_trace, "Link", None)
    _OtelSpanContext = getattr(_otel_trace, "SpanContext", None)
    _OtelTraceFlags = getattr(_otel_trace, "TraceFlags", None)
except Exception:  # pragma: no cover - otel optional
    _otel_trace = None
    _OtelLink = None
    _OtelSpanContext = None
    _OtelTraceFlags = None

# Metrics (optional)
_otel_metrics: Any = None
_meter: Any = None
_counter_recorded: Any = None
_counter_replayed: Any = None
try:  # pragma: no cover - otel optional
    _otel_metrics = importlib.import_module("opentelemetry.metrics")
except Exception:  # pragma: no cover
    _otel_metrics = None


def _get_tracer() -> Any:
    if _otel_trace is None:  # pragma: no cover - otel optional
        return None
    try:
        return _otel_trace.get_tracer("timewarp")
    except Exception:  # pragma: no cover - defensive
        return None


def configure(
    *,
    provider: Any | None = None,
    exporter: Any | None = None,
    sampling: str | None = None,
    redact_payloads: bool = False,  # reserved for future use
) -> Any | None:
    """Configure OpenTelemetry tracer provider/exporter and initialize metrics counters.

    Best-effort, no-op when OpenTelemetry SDK is unavailable.
    """
    # Tracing provider/exporter
    try:  # pragma: no cover - otel optional
        if _otel_trace is None:
            return None
        tp: Any | None = provider
        if tp is None:
            sdk_trace = importlib.import_module("opentelemetry.sdk.trace")
            tp = sdk_trace.TracerProvider()
        # Sampling option (basic support)
        if sampling is None:
            sampling = os.environ.get("TIMEWARP_OTEL_SAMPLING")
        if sampling:
            try:
                sdk_trace_sampl = importlib.import_module("opentelemetry.sdk.trace.sampling")
                # Normalize common aliases
                s = sampling.strip().lower()
                if s in {"on", "always_on", "always-on", "true", "yes", "1"}:
                    sampler = sdk_trace_sampl.ALWAYS_ON
                elif s.startswith("ratio(") and s.endswith(")"):
                    rate = float(s[len("ratio(") : -1])
                    sampler = sdk_trace_sampl.TraceIdRatioBased(rate)
                elif s == "parent":
                    # Parent-based sampling can be approximated by keeping the provider default
                    sampler = None
                else:
                    sampler = None
                if sampler is not None and tp is not None:
                    # Assign sampler when tracer provider supports it
                    try:
                        tp.sampler = sampler
                    except Exception:
                        pass
            except Exception:
                pass
        if exporter is not None and tp is not None:
            sdk_trace = importlib.import_module("opentelemetry.sdk.trace")
            proc_mod = importlib.import_module("opentelemetry.sdk.trace.export")
            bsp_cls = proc_mod.BatchSpanProcessor
            processor = bsp_cls(exporter)
            try:
                tp.add_span_processor(processor)
            except Exception:
                pass
        set_tp = getattr(_otel_trace, "set_tracer_provider", None)
        if set_tp and tp is not None:
            set_tp(tp)
    except Exception:
        tp = None

    # Metrics counters
    global _meter, _counter_recorded, _counter_replayed
    try:  # pragma: no cover - otel optional
        if _otel_metrics is not None and _meter is None:
            get_meter = getattr(_otel_metrics, "get_meter", None)
            if callable(get_meter):
                _meter = get_meter("timewarp")
                _counter_recorded = _meter.create_counter(
                    name="tw.events.recorded", unit="1", description="Recorded events"
                )
                _counter_replayed = _meter.create_counter(
                    name="tw.events.replayed", unit="1", description="Replayed events"
                )
    except Exception:
        _meter = None
        _counter_recorded = None
        _counter_replayed = None
    # Single informational line when tracing is enabled
    try:
        if tp is not None:
            prov_name = type(tp).__name__ if tp is not None else "-"
            log_warn_once(
                "otel.enabled", None, {"provider": prov_name, "sampling": sampling or "default"}
            )
    except Exception:
        pass
    return tp


def _span_name(kind: str, action_type: str, actor: str) -> str:
    # Span names keep the project namespace; attributes use tw.* keys.
    return f"timewarp.{kind}.{action_type}.{actor}"


@contextmanager
def record_event_span(ev: Any) -> Iterator[tuple[str | None, str | None]]:
    """Create an OTel span for a recorded event; returns (trace_id_hex, span_id_hex).

    Best-effort and no-op when OpenTelemetry is not available.
    """
    tracer = _get_tracer()
    trace_id_hex: str | None = None
    span_id_hex: str | None = None
    if tracer is None:
        yield (trace_id_hex, span_id_hex)
        return
    name = _span_name("record", ev.action_type.value, ev.actor)
    try:
        with tracer.start_as_current_span(name) as span:
            try:
                _set_span_attrs(span, ev, replay=False)
                ctx = span.get_span_context()
                trace_id_hex = _fmt_trace_id(ctx.trace_id)
                span_id_hex = _fmt_span_id(ctx.span_id)
            except Exception:  # pragma: no cover - defensive
                pass
            try:
                if _counter_recorded is not None:
                    _counter_recorded.add(1, {"action_type": ev.action_type.value})
            except Exception:
                pass
            yield (trace_id_hex, span_id_hex)
    except Exception:  # pragma: no cover - defensive
        yield (trace_id_hex, span_id_hex)


@contextmanager
def replay_span_for_event(ev: Any) -> Iterator[None]:
    """Create an OTel span for replay of an event and link to original span if present."""
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    name = _span_name("replay", ev.action_type.value, ev.actor)
    links: list[Any] = []
    try:
        mm = ev.model_meta or {}
        t_hex = mm.get("otel_trace_id") if isinstance(mm, dict) else None
        s_hex = mm.get("otel_span_id") if isinstance(mm, dict) else None
        if _OtelSpanContext is not None and t_hex and s_hex:
            try:
                t_id = int(t_hex, 16)
                s_id = int(s_hex, 16)
                flags = _OtelTraceFlags(0x01) if _OtelTraceFlags is not None else None  # sampled
                ctx = _OtelSpanContext(
                    trace_id=t_id, span_id=s_id, is_remote=True, trace_flags=flags
                )
                if _OtelLink is not None:
                    links.append(_OtelLink(ctx))
            except Exception:  # pragma: no cover
                pass
    except Exception:
        pass

    try:
        if links:
            with tracer.start_as_current_span(name, links=links):
                try:
                    _set_span_attrs(None, ev, replay=True, into_current=True)
                except Exception:  # pragma: no cover
                    pass
                yield None
        else:
            with tracer.start_as_current_span(name):
                try:
                    _set_span_attrs(None, ev, replay=True, into_current=True)
                except Exception:  # pragma: no cover
                    pass
                yield None
    except Exception:  # pragma: no cover - defensive
        yield None
    # Increment counter outside span to be resilient
    try:
        if _counter_replayed is not None:
            _counter_replayed.add(1, {"action_type": ev.action_type.value})
    except Exception:
        pass


def _fmt_trace_id(i: int) -> str:
    return f"{i:032x}"


def _fmt_span_id(i: int) -> str:
    return f"{i:016x}"


def _set_span_attrs(span: Any, ev: Any, *, replay: bool, into_current: bool = False) -> None:
    # Attempt to set attributes on passed span, else on current span
    if _otel_trace is None:  # pragma: no cover
        return
    try:
        tgt = span
        if tgt is None and into_current:
            tgt = _otel_trace.get_current_span()
        if tgt is None:
            return
        # Attributes
        attrs = {
            "tw.run_id": str(ev.run_id),
            "tw.step": int(ev.step),
            "tw.action_type": ev.action_type.value,
            "tw.actor": ev.actor,
            "tw.replay": bool(replay),
        }
        if getattr(ev, "labels", None):
            lab = ev.labels
            ns = lab.get("namespace")
            tid = lab.get("thread_id")
            cp = lab.get("checkpoint_id")
            anch = lab.get("anchor_id")
            branch_of = lab.get("branch_of")
            if ns is not None:
                attrs["tw.namespace"] = ns
            if tid is not None:
                attrs["tw.thread_id"] = tid
            if cp is not None:
                attrs["tw.checkpoint_id"] = cp
            if anch is not None:
                attrs["tw.anchor_id"] = anch
            if branch_of is not None:
                attrs["tw.branch_of"] = branch_of
        if getattr(ev, "hashes", None):
            out_h = ev.hashes.get("output")
            st_h = ev.hashes.get("state")
            pr_h = ev.hashes.get("prompt")
            if out_h:
                attrs["tw.hash.output"] = out_h
            if st_h:
                attrs["tw.hash.state"] = st_h
            if pr_h:
                attrs["tw.hash.prompt"] = pr_h
        try:
            # set_attributes exists on Span in otel API
            tgt.set_attributes(attrs)
        except Exception:
            for k, v in attrs.items():
                try:
                    tgt.set_attribute(k, v)
                except Exception:  # pragma: no cover
                    pass
    except Exception:  # pragma: no cover
        pass
