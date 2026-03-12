import torch
from .interface import (
    CpuArchEnum as CpuArchEnum,
    Platform as Platform,
    PlatformEnum as PlatformEnum,
)
from _typeshed import Incomplete
from dataclasses import dataclass
from vllm import envs as envs
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.v1.attention.backend import is_quantized_kv_cache as is_quantized_kv_cache
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)
from vllm.v1.attention.selector import (
    AttentionSelectorConfig as AttentionSelectorConfig,
)

logger: Incomplete

def get_max_threads(pid: int = 0): ...
@dataclass
class LogicalCPUInfo:
    id: int = ...
    physical_core: int = ...
    numa_node: int = ...
    @staticmethod
    def json_decoder(obj_dict: dict): ...

class CpuPlatform(Platform):
    device_name: str
    device_type: str
    dispatch_key: str
    dist_backend: str
    device_control_env_var: str
    @property
    def supported_dtypes(self) -> list[torch.dtype]: ...
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str: ...
    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> str: ...
    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int: ...
    @classmethod
    def set_device(cls, device: torch.device) -> None: ...
    @classmethod
    def inference_mode(cls): ...
    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None: ...
    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None: ...
    @classmethod
    def get_allowed_cpu_core_node_list(
        cls,
    ) -> tuple[list[int], list[LogicalCPUInfo]]: ...
    @classmethod
    def discover_numa_topology(cls) -> list[list[int]]: ...
    @classmethod
    def is_pin_memory_available(cls) -> bool: ...
    @classmethod
    def get_punica_wrapper(cls) -> str: ...
    @classmethod
    def get_device_communicator_cls(cls) -> str: ...
    @classmethod
    def supports_structured_output(cls) -> bool: ...
    @classmethod
    def opaque_attention_op(cls) -> bool: ...
    @classmethod
    def support_hybrid_kv_cache(cls) -> bool: ...
    @classmethod
    def import_kernels(cls) -> None: ...
