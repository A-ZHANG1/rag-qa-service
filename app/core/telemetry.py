"""OpenTelemetry tracing setup for the RAG service.

Defaults to a zero-config ``ConsoleSpanExporter`` so tracing works out of
the box with no external collector -- consistent with the project's
"free/local by default" philosophy (Ollama, embedded ChromaDB, DuckDuckGo).
Set ``OTEL_EXPORTER_OTLP_ENDPOINT`` to export to a real collector (Jaeger,
Tempo, etc.) instead; the OTLP exporter dependency is already in
requirements.txt but unused until that env var is set.
"""

from __future__ import annotations

import os

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Tracer

_initialized = False


def setup_telemetry(service_name: str = "rag-qa-service") -> None:
    """Initialize the global TracerProvider. Idempotent -- safe to call more than once
    (e.g. once from app startup, once from a test fixture).

    :param service_name: Value for the ``service.name`` resource attribute.
    """
    global _initialized
    if _initialized:
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        # Import deferred: the grpc exporter pulls in grpcio, only needed on this path.
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        # Zero-config default: print spans to stdout as they finish, no collector required.
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _initialized = True


def get_tracer(name: str = "rag_qa_service") -> Tracer:
    """Return a tracer. Safe to call before :func:`setup_telemetry` -- the
    global API returns a no-op tracer until a provider is registered."""
    return trace.get_tracer(name)
