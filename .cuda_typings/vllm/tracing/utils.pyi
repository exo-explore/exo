from _typeshed import Incomplete
from collections.abc import Mapping
from vllm.logger import init_logger as init_logger
from vllm.utils.func_utils import run_once as run_once

logger: Incomplete
TRACE_HEADERS: Incomplete

class SpanAttributes:
    GEN_AI_USAGE_COMPLETION_TOKENS: str
    GEN_AI_USAGE_PROMPT_TOKENS: str
    GEN_AI_REQUEST_MAX_TOKENS: str
    GEN_AI_REQUEST_TOP_P: str
    GEN_AI_REQUEST_TEMPERATURE: str
    GEN_AI_RESPONSE_MODEL: str
    GEN_AI_REQUEST_ID: str
    GEN_AI_REQUEST_N: str
    GEN_AI_USAGE_NUM_SEQUENCES: str
    GEN_AI_LATENCY_TIME_IN_QUEUE: str
    GEN_AI_LATENCY_TIME_TO_FIRST_TOKEN: str
    GEN_AI_LATENCY_E2E: str
    GEN_AI_LATENCY_TIME_IN_SCHEDULER: str
    GEN_AI_LATENCY_TIME_IN_MODEL_FORWARD: str
    GEN_AI_LATENCY_TIME_IN_MODEL_EXECUTE: str
    GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL: str
    GEN_AI_LATENCY_TIME_IN_MODEL_DECODE: str
    GEN_AI_LATENCY_TIME_IN_MODEL_INFERENCE: str

class LoadingSpanAttributes:
    CODE_NAMESPACE: str
    CODE_FUNCTION: str
    CODE_FILEPATH: str
    CODE_LINENO: str

def contains_trace_headers(headers: Mapping[str, str]) -> bool: ...
def extract_trace_headers(headers: Mapping[str, str]) -> Mapping[str, str]: ...
@run_once
def log_tracing_disabled_warning() -> None: ...
