import torch
from .interface import (
    DeviceCapability as DeviceCapability,
    Platform as Platform,
    PlatformEnum as PlatformEnum,
)
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from datetime import timedelta
from functools import cache
from torch.distributed import PrefixStore as PrefixStore, ProcessGroup
from vllm.config import VllmConfig as VllmConfig
from vllm.config.cache import CacheDType as CacheDType
from vllm.logger import init_logger as init_logger
from vllm.utils.import_utils import import_pynvml as import_pynvml
from vllm.utils.torch_utils import (
    cuda_device_count_stateless as cuda_device_count_stateless,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)
from vllm.v1.attention.selector import (
    AttentionSelectorConfig as AttentionSelectorConfig,
)

logger: Incomplete
pynvml: Incomplete

def with_nvml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]: ...

class CudaPlatformBase(Platform):
    device_name: str
    device_type: str
    dispatch_key: str
    ray_device_key: str
    dist_backend: str
    device_control_env_var: str
    ray_noset_device_env_vars: list[str]
    @property
    def supported_dtypes(self) -> list[torch.dtype]: ...
    @classmethod
    def set_device(cls, device: torch.device) -> None: ...
    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None: ...
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str: ...
    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int: ...
    @classmethod
    def is_fully_connected(cls, device_ids: list[int]) -> bool: ...
    @classmethod
    def log_warnings(cls) -> None: ...
    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None: ...
    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float: ...
    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", tuple[int, list[str]]],
    ]: ...
    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum | None,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> str: ...
    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]: ...
    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: AttentionBackendEnum | None = None,
    ) -> AttentionBackendEnum: ...
    @classmethod
    def get_punica_wrapper(cls) -> str: ...
    @classmethod
    def get_device_communicator_cls(cls) -> str: ...
    @classmethod
    def supports_fp8(cls) -> bool: ...
    @classmethod
    def use_custom_allreduce(cls) -> bool: ...
    @classmethod
    def opaque_attention_op(cls) -> bool: ...
    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str: ...
    @classmethod
    def stateless_init_device_torch_dist_pg(
        cls,
        backend: str,
        prefix_store: PrefixStore,
        group_rank: int,
        group_size: int,
        timeout: timedelta,
    ) -> ProcessGroup: ...
    @classmethod
    def device_count(cls) -> int: ...
    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype): ...
    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None: ...
    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None: ...
    @classmethod
    def support_hybrid_kv_cache(cls) -> bool: ...
    @classmethod
    def support_static_graph_mode(cls) -> bool: ...
    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int: ...
    @classmethod
    def use_custom_op_collectives(cls) -> bool: ...

class NvmlCudaPlatform(CudaPlatformBase):
    @classmethod
    @cache
    @with_nvml_context
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None: ...
    @classmethod
    @with_nvml_context
    def has_device_capability(
        cls, capability: tuple[int, int] | int, device_id: int = 0
    ) -> bool: ...
    @classmethod
    @with_nvml_context
    def get_device_name(cls, device_id: int = 0) -> str: ...
    @classmethod
    @with_nvml_context
    def get_device_uuid(cls, device_id: int = 0) -> str: ...
    @classmethod
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int: ...
    @classmethod
    @with_nvml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool: ...
    @classmethod
    @with_nvml_context
    def log_warnings(cls) -> None: ...

class NonNvmlCudaPlatform(CudaPlatformBase):
    @classmethod
    @cache
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability: ...
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str: ...
    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int: ...
    @classmethod
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool: ...

nvml_available: bool
CudaPlatform: Incomplete
