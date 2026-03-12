from .otel import (
    SpanKind as SpanKind,
    extract_trace_context as extract_trace_context,
    otel_import_error_traceback as otel_import_error_traceback,
)
from .utils import (
    SpanAttributes as SpanAttributes,
    contains_trace_headers as contains_trace_headers,
    extract_trace_headers as extract_trace_headers,
    log_tracing_disabled_warning as log_tracing_disabled_warning,
)
from collections.abc import Callable
from typing import Any, TypeAlias

__all__ = [
    "instrument",
    "instrument_manual",
    "init_tracer",
    "maybe_init_worker_tracer",
    "is_tracing_available",
    "SpanAttributes",
    "SpanKind",
    "extract_trace_context",
    "extract_trace_headers",
    "log_tracing_disabled_warning",
    "contains_trace_headers",
    "otel_import_error_traceback",
]

BackendAvailableFunc: TypeAlias = Callable[[], bool]
InstrumentFunc: TypeAlias = Callable[..., Any]
InstrumentManualFunc: TypeAlias = Callable[..., Any]
InitTracerFunc: TypeAlias = Callable[..., Any]
InitWorkerTracerFunc: TypeAlias = Callable[..., Any]

def init_tracer(
    instrumenting_module_name: str,
    otlp_traces_endpoint: str,
    extra_attributes: dict[str, str] | None = None,
): ...
def maybe_init_worker_tracer(
    instrumenting_module_name: str, process_kind: str, process_name: str
): ...
def instrument(
    obj: Callable | None = None,
    *,
    span_name: str = "",
    attributes: dict[str, str] | None = None,
    record_exception: bool = True,
): ...
def instrument_manual(
    span_name: str,
    start_time: int,
    end_time: int | None = None,
    attributes: dict[str, Any] | None = None,
    context: Any = None,
    kind: Any = None,
): ...
def is_tracing_available() -> bool: ...
