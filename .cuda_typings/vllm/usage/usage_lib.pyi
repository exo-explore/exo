from _typeshed import Incomplete
from enum import Enum
from typing import Any
from vllm.connections import global_http_connection as global_http_connection
from vllm.logger import init_logger as init_logger
from vllm.utils.platform_utils import (
    cuda_get_device_properties as cuda_get_device_properties,
)
from vllm.utils.torch_utils import (
    cuda_device_count_stateless as cuda_device_count_stateless,
)

logger: Incomplete

def set_runtime_usage_data(key: str, value: str | int | bool) -> None: ...
def is_usage_stats_enabled(): ...

class UsageContext(str, Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    OPENAI_BATCH_RUNNER = "OPENAI_BATCH_RUNNER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"

class UsageMessage:
    uuid: Incomplete
    provider: str | None
    num_cpu: int | None
    cpu_type: str | None
    cpu_family_model_stepping: str | None
    total_memory: int | None
    architecture: str | None
    platform: str | None
    cuda_runtime: str | None
    gpu_count: int | None
    gpu_type: str | None
    gpu_memory_per_device: int | None
    env_var_json: str | None
    model_architecture: str | None
    vllm_version: str | None
    context: str | None
    log_time: int | None
    source: str | None
    def __init__(self) -> None: ...
    def report_usage(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any] | None = None,
    ) -> None: ...

usage_message: Incomplete
