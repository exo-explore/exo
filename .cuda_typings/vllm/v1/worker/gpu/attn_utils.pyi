import torch
from collections.abc import Sequence
from typing import Any
from vllm.config import (
    VllmConfig as VllmConfig,
    get_layers_from_vllm_config as get_layers_from_vllm_config,
)
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    CommonAttentionMetadata as CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    KVCacheConfig as KVCacheConfig,
    KVCacheSpec as KVCacheSpec,
    UniformTypeKVCacheSpecs as UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.utils import (
    AttentionGroup as AttentionGroup,
    bind_kv_cache as bind_kv_cache,
)

def get_kv_cache_spec(vllm_config: VllmConfig) -> dict[str, KVCacheSpec]: ...
def init_attn_backend(
    kv_cache_config: KVCacheConfig, vllm_config: VllmConfig, device: torch.device
): ...
def init_kv_cache(
    runner_kv_caches: list[torch.Tensor],
    forward_context: dict[str, Any],
    kv_cache_config: KVCacheConfig,
    attn_backends: dict[str, AttentionBackend],
    device: torch.device,
) -> dict[str, torch.Tensor]: ...
def build_slot_mappings_by_layer(
    slot_mappings: torch.Tensor, kv_cache_config: KVCacheConfig
) -> dict[str, torch.Tensor]: ...
def build_attn_metadata(
    attn_groups: list[list[AttentionGroup]],
    num_reqs: int,
    num_tokens: int,
    query_start_loc_gpu: torch.Tensor,
    query_start_loc_cpu: torch.Tensor,
    max_query_len: int,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    block_tables: Sequence[torch.Tensor],
    slot_mappings: torch.Tensor,
    kv_cache_config: KVCacheConfig,
    dcp_local_seq_lens: torch.Tensor | None = None,
) -> dict[str, Any]: ...
