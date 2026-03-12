import torch
from _typeshed import Incomplete
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from vllm.config import CacheConfig as CacheConfig, VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention import Attention as Attention
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings as MultiModalEmbeddings,
)
from vllm.model_executor.models.utils import extract_layer_index as extract_layer_index
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import (
    largest_power_of_2_divisor as largest_power_of_2_divisor,
)
from vllm.utils.mem_utils import (
    MemorySnapshot as MemorySnapshot,
    format_gib as format_gib,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    MultipleOf as MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    EncoderOnlyAttentionSpec as EncoderOnlyAttentionSpec,
    FullAttentionSpec as FullAttentionSpec,
    KVCacheConfig as KVCacheConfig,
    KVCacheGroupSpec as KVCacheGroupSpec,
    KVCacheSpec as KVCacheSpec,
    MambaSpec as MambaSpec,
    UniformTypeKVCacheSpecs as UniformTypeKVCacheSpecs,
)

logger: Incomplete

class KVBlockZeroer:
    device: Incomplete
    pin_memory: Incomplete
    def __init__(self, device: torch.device, pin_memory: bool) -> None: ...
    def init_meta(
        self,
        attn_groups_iter: Iterable["AttentionGroup"],
        kernel_block_sizes: list[int],
        cache_dtype: str,
        runner_only_attn_layers: set[str],
        static_forward_context: dict[str, Any],
    ) -> None: ...
    def zero_block_ids(self, block_ids: list[int]) -> None: ...

@dataclass
class AttentionGroup:
    backend: type[AttentionBackend]
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec
    kv_cache_group_id: int
    metadata_builders: list[AttentionMetadataBuilder] = field(
        default_factory=Incomplete
    )
    def create_metadata_builders(
        self,
        vllm_config,
        device,
        kernel_block_size: int | None = None,
        num_metadata_builders: int = 1,
    ): ...
    def get_metadata_builder(self, ubatch_id: int = 0) -> AttentionMetadataBuilder: ...

def select_common_block_size(
    kv_manager_block_size: int, backends: list[type[AttentionBackend]]
) -> int: ...
def prepare_kernel_block_sizes(
    kv_cache_config: KVCacheConfig, attn_groups: list[list[AttentionGroup]]
) -> list[int]: ...
def sanity_check_mm_encoder_outputs(
    mm_embeddings: MultiModalEmbeddings, expected_num_items: int
) -> None: ...
def request_memory(init_snapshot: MemorySnapshot, cache_config: CacheConfig) -> int: ...
def add_kv_sharing_layers_to_kv_cache_groups(
    shared_kv_cache_layers: dict[str, str],
    kv_cache_groups: list[KVCacheGroupSpec],
    runner_only_attn_layers: set[str] | None = None,
) -> None: ...
def bind_kv_cache(
    kv_caches: dict[str, torch.Tensor],
    forward_context: dict[str, Attention],
    runner_kv_caches: list[torch.Tensor],
    num_attn_module: int = 1,
) -> None: ...
def is_residual_scattered_for_sp(
    vllm_config: VllmConfig, num_input_tokens: int
) -> bool: ...
