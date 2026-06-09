"""
Prometheus metrics for exo inference monitoring.

Exposes metrics for request tracking, token generation, latency,
and RKLLM-specific performance data.
"""

from prometheus_client import Counter, Gauge, Histogram, Info

# =============================================================================
# Request Metrics
# =============================================================================

REQUESTS_TOTAL = Counter(
  'exo_requests_total',
  'Total number of inference requests',
  ['model', 'status']
)

REQUESTS_IN_PROGRESS = Gauge(
  'exo_requests_in_progress',
  'Number of requests currently being processed'
)

# =============================================================================
# Token Metrics
# =============================================================================

TOKENS_GENERATED = Counter(
  'exo_tokens_generated_total',
  'Total number of tokens generated',
  ['model']
)

PROMPT_TOKENS = Counter(
  'exo_prompt_tokens_total',
  'Total number of prompt tokens processed',
  ['model']
)

# =============================================================================
# Latency Metrics
# =============================================================================

REQUEST_LATENCY = Histogram(
  'exo_request_duration_seconds',
  'Request latency in seconds',
  buckets=[.1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120, 300]
)

FIRST_TOKEN_LATENCY = Histogram(
  'exo_first_token_latency_seconds',
  'Time to first token in seconds',
  buckets=[.05, .1, .25, .5, 1, 2, 5, 10, 30, 60]
)

TOKENS_PER_SECOND = Gauge(
  'exo_tokens_per_second',
  'Current token generation rate',
  ['model']
)

# =============================================================================
# System Metrics
# =============================================================================

NODES_ACTIVE = Gauge(
  'exo_nodes_active',
  'Number of active nodes in cluster'
)

MODEL_INFO = Info(
  'exo_model',
  'Information about the currently loaded model'
)

INFERENCE_ENGINE_INFO = Info(
  'exo_inference_engine',
  'Information about the inference engine'
)

# =============================================================================
# RKLLM-Specific Metrics (conditionally imported from rkllm module)
# =============================================================================

# Import RKLLM metrics from the rkllm module for backward compatibility
try:
  from exo.inference.rkllm.metrics import (
    RKLLM_SERVER_UP,
    RKLLM_INFERENCE_SECONDS,
    RKLLM_MODEL_LOAD_SECONDS,
    RKLLM_TOKENS_CACHED,
    RKLLM_HTTP_REQUESTS,
  )
except ImportError:
  # RKLLM module not available - create placeholder metrics
  RKLLM_SERVER_UP = Gauge('rkllm_server_up', 'RKLLAMA server availability (1=up, 0=down)')
  RKLLM_INFERENCE_SECONDS = Histogram('rkllm_inference_duration_seconds', 'RKLLM inference call duration', buckets=[.01, .05, .1, .25, .5, 1, 2, 5, 10, 30])
  RKLLM_MODEL_LOAD_SECONDS = Histogram('rkllm_model_load_duration_seconds', 'Time to load model on RKLLM server', buckets=[1, 5, 10, 30, 60, 120, 300])
  RKLLM_TOKENS_CACHED = Gauge('rkllm_tokens_cached', 'Number of tokens currently cached per request', ['request_id'])
  RKLLM_HTTP_REQUESTS = Counter('rkllm_http_requests_total', 'Total HTTP requests to RKLLAMA server', ['endpoint', 'status'])

# =============================================================================
# Error Metrics
# =============================================================================

ERRORS_TOTAL = Counter(
  'exo_errors_total',
  'Total number of errors',
  ['type', 'model']
)
