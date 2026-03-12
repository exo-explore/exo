import abc
import torch
from dataclasses import dataclass
from vllm.config import VllmConfig as VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    CommonAttentionMetadata as CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    mamba_get_block_table_tensor as mamba_get_block_table_tensor,
    split_decodes_and_prefills as split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    MambaSpec as MambaSpec,
)

class LinearAttentionBackend(AttentionBackend, metaclass=abc.ABCMeta):
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_builder_cls() -> type["LinearAttentionMetadataBuilder"]: ...

@dataclass
class LinearAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    state_indices_tensor: torch.Tensor

class LinearAttentionMetadataBuilder(AttentionMetadataBuilder[LinearAttentionMetadata]):
    reorder_batch_threshold: int
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> LinearAttentionMetadata: ...
