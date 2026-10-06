from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)

_SERVICE_NAME = "agentic-rag"

# Held here rather than installed globally: OpenTelemetry's global provider
# can only be set once per process, which would make it impossible to swap
# in an in-memory exporter under test. Until configure_tracing() runs (e.g.
# in the CLI), get_tracer() returns the API's no-op tracer.
_provider: TracerProvider | None = None


def configure_tracing(exporter: SpanExporter | None = None) -> None:
    """Install a tracer provider. Defaults to one-line JSON spans on stdout,
    next to the structured logs (which carry the same trace_id)."""
    global _provider
    exporter = exporter or ConsoleSpanExporter(
        formatter=lambda span: span.to_json(indent=None) + "\n"
    )
    _provider = TracerProvider(resource=Resource.create({"service.name": _SERVICE_NAME}))
    _provider.add_span_processor(SimpleSpanProcessor(exporter))


def get_tracer() -> trace.Tracer:
    if _provider is None:
        return trace.get_tracer(_SERVICE_NAME)
    return _provider.get_tracer(_SERVICE_NAME)


def current_trace_ids() -> tuple[str, str] | None:
    """(trace_id, span_id) of the active span as hex, or None if there isn't one."""
    context = trace.get_current_span().get_span_context()
    if not context.is_valid:
        return None
    return format(context.trace_id, "032x"), format(context.span_id, "016x")
