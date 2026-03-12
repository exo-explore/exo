import torch
from .interface import (
    DeviceCapability as DeviceCapability,
    Platform as Platform,
    PlatformEnum as PlatformEnum,
)
from _typeshed import Incomplete
from datetime import timedelta
from functools import cache
from torch.distributed import PrefixStore as PrefixStore, ProcessGroup
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
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

def with_amdsmi_context(fn): ...
def on_gfx1x() -> bool: ...
def on_mi3xx() -> bool: ...
def on_gfx9() -> bool: ...
def on_gfx942() -> bool: ...
def on_gfx950() -> bool: ...
@cache
def use_rocm_custom_paged_attention(
    qtype: torch.dtype,
    head_size: int,
    block_size: int,
    gqa_ratio: int,
    max_seq_len: int,
    sliding_window: int,
    kv_cache_dtype: str,
    alibi_slopes: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> bool: ...
@cache
def flash_attn_triton_available() -> bool: ...

class RocmPlatform(Platform):
    device_name: str
    device_type: str
    dispatch_key: str
    ray_device_key: str
    dist_backend: str
    device_control_env_var: str
    ray_noset_device_env_vars: list[str]
    supported_quantization: list[str]
    @classmethod
    def import_kernels(cls) -> None: ...
    @classmethod
    def get_valid_backends(
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: AttentionSelectorConfig,
        num_heads: int | None = None,
    ) -> tuple[
        list[tuple["AttentionBackendEnum", int]],
        dict["AttentionBackendEnum", list[str]],
    ]: ...
    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
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
    def set_device(cls, device: torch.device) -> None: ...
    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None: ...
    @classmethod
    @with_amdsmi_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool: ...
    @classmethod
    @with_amdsmi_context
    def get_device_name(cls, device_id: int = 0) -> str: ...
    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int: ...
    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: VllmConfig) -> None: ...
    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None: ...
    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None: ...
    @classmethod
    def verify_model_arch(cls, model_arch: str) -> None: ...
    @classmethod
    def verify_quantization(cls, quant: str) -> None: ...
    @classmethod
    def get_punica_wrapper(cls) -> str: ...
    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float: ...
    @classmethod
    def get_device_communicator_cls(cls) -> str: ...
    @classmethod
    def supports_mx(cls) -> bool: ...
    @classmethod
    def supports_fp8(cls) -> bool: ...
    @classmethod
    def is_fp8_fnuz(cls) -> bool: ...
    @classmethod
    def fp8_dtype(cls) -> torch.dtype: ...
    @classmethod
    def use_custom_allreduce(cls) -> bool: ...
    @classmethod
    def opaque_attention_op(cls) -> bool: ...
    @classmethod
    def is_navi(cls) -> bool: ...
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
    def support_hybrid_kv_cache(cls) -> bool: ...
    @classmethod
    def support_static_graph_mode(cls) -> bool: ...
    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int: ...
    @classmethod
    def use_custom_op_collectives(cls) -> bool: ...
