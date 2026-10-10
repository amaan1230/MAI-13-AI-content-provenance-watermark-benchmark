"""OpenTelemetry tracing (optional).

Spans are always created through the OTel API, which is a no-op until a provider is
configured. Set OTEL_ENABLED=true to export spans: to an OTLP endpoint if
OTEL_EXPORTER_OTLP_ENDPOINT is set and the OTLP exporter package is installed,
otherwise to the console.
"""

from __future__ import annotations

import os

from backend.core.logging import get_logger

log = get_logger(__name__)
_configured = False


def setup_tracing(service_name: str = "mai13-api") -> bool:
    global _configured
    if _configured or os.getenv("OTEL_ENABLED", "false").lower() != "true":
        return False
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        log.warning("OTEL_ENABLED=true but opentelemetry-sdk is not installed")
        return False
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    exporter = None
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter()
        except ImportError:
            log.warning("OTLP exporter not installed; falling back to console exporter")
    provider.add_span_processor(BatchSpanProcessor(exporter or ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True
    return True


def get_tracer(name: str = "mai13"):
    from opentelemetry import trace

    return trace.get_tracer(name)
