from _typeshed import Incomplete
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from opentelemetry.context.context import Context
from opentelemetry.trace import Tracer
from typing import Any
from vllm.logger import init_logger as init_logger
from vllm.tracing.utils import (
    LoadingSpanAttributes as LoadingSpanAttributes,
    TRACE_HEADERS as TRACE_HEADERS,
)

logger: Incomplete
otel_import_error_traceback: Incomplete
Context = Any
Tracer = Any
SpanKind = Any

def is_otel_available() -> bool: ...
def init_otel_tracer(
    instrumenting_module_name: str,
    otlp_traces_endpoint: str,
    extra_attributes: dict[str, str] | None = None,
) -> Tracer: ...
def get_span_exporter(endpoint): ...
def init_otel_worker_tracer(
    instrumenting_module_name: str, process_kind: str, process_name: str
) -> Tracer: ...
def extract_trace_context(headers: Mapping[str, str] | None) -> Context | None: ...
def instrument_otel(func, span_name, attributes, record_exception): ...
def manual_instrument_otel(
    span_name: str,
    start_time: int,
    end_time: int | None = None,
    attributes: dict[str, Any] | None = None,
    context: Context | None = None,
    kind: Any = None,
): ...
@contextmanager
def propagate_trace_to_env() -> Generator[None]: ...
