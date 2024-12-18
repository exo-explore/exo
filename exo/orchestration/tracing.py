from dataclasses import dataclass
from typing import Dict, Optional, Any
from opentelemetry import trace, context
from opentelemetry.trace import Status, StatusCode, SpanContext
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from contextlib import contextmanager
import time
from threading import Lock

@dataclass
class TraceContext:
  request_id: str
  sequence_number: int
  current_span: Optional[trace.Span] = None
  trace_parent: Optional[str] = None
  token_group_span: Optional[trace.Span] = None
  token_count: int = 0
  token_group_size: int = 10  # Default group size
  request_span: Optional[trace.Span] = None  # Track the main request span

class Tracer:
  def __init__(self):
    self.tracer = trace.get_tracer("exo")
    self.contexts: Dict[str, TraceContext] = {}
    self._lock = Lock()
    self.propagator = TraceContextTextMapPropagator()
    
  def get_context(self, request_id: str) -> Optional[TraceContext]:
    with self._lock:
      return self.contexts.get(request_id)

  def set_context(self, request_id: str, context: TraceContext):
    with self._lock:
      self.contexts[request_id] = context

  def inject_context(self, span: trace.Span) -> str:
    """Inject current span context into carrier for propagation"""
    carrier = {}
    ctx = trace.set_span_in_context(span)
    self.propagator.inject(carrier, context=ctx)
    return carrier.get("traceparent", "")

  def extract_context(self, trace_parent: str) -> Optional[context.Context]:
    """Extract span context from carrier"""
    if not trace_parent:
      return None
    carrier = {"traceparent": trace_parent}
    return self.propagator.extract(carrier)

  def create_context_from_parent(self, request_id: str, trace_parent: str, sequence_number: int = 0) -> TraceContext:
    """Create a new context with the given trace parent"""
    parent_ctx = self.extract_context(trace_parent)
    if parent_ctx:
      # Create a new request span that links to the parent context
      request_span = self.tracer.start_span(
        "request",
        context=parent_ctx,
        attributes={
          "request_id": request_id,
          "sequence_number": sequence_number
        }
      )
      return TraceContext(
        request_id=request_id,
        sequence_number=sequence_number,
        request_span=request_span,
        current_span=request_span,
        trace_parent=trace_parent
      )
    return TraceContext(request_id=request_id, sequence_number=sequence_number)

  def handle_token(self, context: TraceContext, token: int, is_finished: bool = False):
    """Handle token generation and manage token group spans"""
    context.token_count += 1
    
    # Start a new token group span if needed
    if not context.token_group_span and context.request_span:
      group_number = (context.token_count - 1) // context.token_group_size + 1
      
      # Create token group span as child of request span
      parent_ctx = trace.set_span_in_context(context.request_span)
      context.token_group_span = self.tracer.start_span(
        f"token_group_{group_number}",
        context=parent_ctx,
        attributes={
          "request_id": context.request_id,
          "group.number": group_number,
          "group.start_token": context.token_count,
          "group.max_tokens": context.token_group_size
        }
      )
    
    # Add token to current group span
    if context.token_group_span:
      relative_pos = ((context.token_count - 1) % context.token_group_size) + 1
      context.token_group_span.set_attribute(f"token.{relative_pos}", token)
      context.token_group_span.set_attribute("token.count", relative_pos)
      
      # End current group span if we've reached the group size or if generation is finished
      if context.token_count % context.token_group_size == 0 or is_finished:
        context.token_group_span.set_attribute("token.final_count", relative_pos)
        context.token_group_span.end()
        context.token_group_span = None

  @contextmanager
  def start_span(self, name: str, context: TraceContext, extra_attributes: Optional[Dict[str, Any]] = None):
    """Start a new span with proper parent context"""
    attributes = {
      "request_id": context.request_id,
      "sequence_number": context.sequence_number
    }
    if extra_attributes:
      attributes.update(extra_attributes)
      
    # Use request span as parent if available
    parent_ctx = None
    if context.request_span:
      parent_ctx = trace.set_span_in_context(context.request_span)
    elif context.trace_parent:
      parent_ctx = self.extract_context(context.trace_parent)
      if parent_ctx and not context.request_span:
        # Create a new request span that links to the parent context
        context.request_span = self.tracer.start_span(
          "request",
          context=parent_ctx,
          attributes={
            "request_id": context.request_id,
            "sequence_number": context.sequence_number
          }
        )
        parent_ctx = trace.set_span_in_context(context.request_span)
    elif context.current_span:
      parent_ctx = trace.set_span_in_context(context.current_span)
    
    # Create span with parent context if it exists
    if parent_ctx:
      span = self.tracer.start_span(
        name,
        context=parent_ctx,
        attributes=attributes
      )
    else:
      span = self.tracer.start_span(
        name,
        attributes=attributes
      )
    
    # Update context with current span
    prev_span = context.current_span
    context.current_span = span
    
    try:
      start_time = time.perf_counter()
      yield span
      duration = time.perf_counter() - start_time
      span.set_attribute("duration_s", duration)
      span.set_status(Status(StatusCode.OK))
    except Exception as e:
      span.set_status(Status(StatusCode.ERROR, str(e)))
      raise
    finally:
      span.end()
      context.current_span = prev_span

# Global tracer instance
tracer = Tracer() 