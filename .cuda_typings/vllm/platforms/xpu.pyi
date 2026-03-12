import torch
from .interface import (
    DeviceCapability as DeviceCapability,
    Platform as Platform,
    PlatformEnum as PlatformEnum,
)
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.utils.torch_utils import supports_xpu_graph as supports_xpu_graph
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)
from vllm.v1.attention.selector import (
    AttentionSelectorConfig as AttentionSelectorConfig,
)

logger: Incomplete

class XPUPlatform(Platform):
    device_name: str
    device_type: str
    dispatch_key: str
    ray_device_key: str
    dist_backend: str
    device_control_env_var: str
    @classmethod
    def import_kernels(cls) -> None: ...
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
    def get_device_name(cls, device_id: int = 0) -> str: ...
    @classmethod
    def get_punica_wrapper(cls) -> str: ...
    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int: ...
    @classmethod
    def inference_mode(cls): ...
    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str: ...
    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None: ...
    @classmethod
    def update_block_size_for_backend(cls, vllm_config: VllmConfig) -> None: ...
    @classmethod
    def support_hybrid_kv_cache(cls) -> bool: ...
    @classmethod
    def support_static_graph_mode(cls) -> bool: ...
    @classmethod
    def is_pin_memory_available(cls): ...
    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float: ...
    @classmethod
    def fp8_dtype(cls) -> torch.dtype: ...
    @classmethod
    def is_data_center_gpu(cls) -> bool: ...
    @classmethod
    def get_device_communicator_cls(cls) -> str: ...
    @classmethod
    def device_count(cls) -> int: ...
    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype): ...
    @classmethod
    def opaque_attention_op(cls) -> bool: ...
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
    def num_compute_units(cls, device_id: int = 0) -> int: ...
