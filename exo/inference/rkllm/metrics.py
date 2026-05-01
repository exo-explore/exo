"""
Prometheus metrics for RKLLM inference engine.

These metrics are specific to the RKLLM/RKLLAMA server integration
and track NPU-specific performance characteristics.
"""

from prometheus_client import Counter, Gauge, Histogram

# =============================================================================
# RKLLM Server Metrics
# =============================================================================

RKLLM_SERVER_UP = Gauge(
  'rkllm_server_up',
  'RKLLAMA server availability (1=up, 0=down)'
)

RKLLM_HTTP_REQUESTS = Counter(
  'rkllm_http_requests_total',
  'Total HTTP requests to RKLLAMA server',
  ['endpoint', 'status']
)

# =============================================================================
# RKLLM Inference Metrics
# =============================================================================

RKLLM_INFERENCE_SECONDS = Histogram(
  'rkllm_inference_duration_seconds',
  'RKLLM inference call duration in seconds',
  buckets=[.01, .05, .1, .25, .5, 1, 2, 5, 10, 30]
)

RKLLM_MODEL_LOAD_SECONDS = Histogram(
  'rkllm_model_load_duration_seconds',
  'Time to load model on RKLLM server',
  buckets=[1, 5, 10, 30, 60, 120, 300]
)

# =============================================================================
# RKLLM Token Caching Metrics
# =============================================================================

RKLLM_TOKENS_CACHED = Gauge(
  'rkllm_tokens_cached',
  'Number of tokens currently cached per request',
  ['request_id']
)

RKLLM_CACHE_HITS = Counter(
  'rkllm_cache_hits_total',
  'Number of token cache hits (avoided HTTP calls)'
)

RKLLM_CACHE_MISSES = Counter(
  'rkllm_cache_misses_total',
  'Number of token cache misses (required HTTP calls)'
)


def record_server_status(is_up: bool):
  """Record RKLLAMA server availability status."""
  RKLLM_SERVER_UP.set(1 if is_up else 0)


def record_http_request(endpoint: str, success: bool):
  """Record an HTTP request to RKLLAMA server."""
  status = 'success' if success else 'error'
  RKLLM_HTTP_REQUESTS.labels(endpoint=endpoint, status=status).inc()


def record_inference_time(duration_seconds: float):
  """Record inference call duration."""
  RKLLM_INFERENCE_SECONDS.observe(duration_seconds)


def record_model_load_time(duration_seconds: float):
  """Record model loading duration."""
  RKLLM_MODEL_LOAD_SECONDS.observe(duration_seconds)


def update_tokens_cached(request_id: str, count: int):
  """Update the number of cached tokens for a request."""
  RKLLM_TOKENS_CACHED.labels(request_id=request_id).set(count)


def clear_tokens_cached(request_id: str):
  """Clear the cached tokens metric for a completed request."""
  try:
    RKLLM_TOKENS_CACHED.remove(request_id)
  except KeyError:
    pass  # Label didn't exist


def record_cache_hit():
  """Record a token cache hit."""
  RKLLM_CACHE_HITS.inc()


def record_cache_miss():
  """Record a token cache miss."""
  RKLLM_CACHE_MISSES.inc()
