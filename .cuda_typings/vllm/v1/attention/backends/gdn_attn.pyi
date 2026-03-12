import abc
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.config import VllmConfig as VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    CommonAttentionMetadata as CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import (
    PAD_SLOT_ID as PAD_SLOT_ID,
    compute_causal_conv1d_metadata as compute_causal_conv1d_metadata,
    mamba_get_block_table_tensor as mamba_get_block_table_tensor,
    split_decodes_and_prefills as split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec as AttentionSpec,
    MambaSpec as MambaSpec,
)

class GDNAttentionBackend(AttentionBackend, metaclass=abc.ABCMeta):
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_builder_cls() -> type["GDNAttentionMetadataBuilder"]: ...

@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int
    has_initial_state: torch.Tensor | None = ...
    spec_query_start_loc: torch.Tensor | None = ...
    non_spec_query_start_loc: torch.Tensor | None = ...
    spec_state_indices_tensor: torch.Tensor | None = ...
    non_spec_state_indices_tensor: torch.Tensor | None = ...
    spec_sequence_masks: torch.Tensor | None = ...
    spec_token_indx: torch.Tensor | None = ...
    non_spec_token_indx: torch.Tensor | None = ...
    num_accepted_tokens: torch.Tensor | None = ...
    nums_dict: dict | None = ...
    batch_ptr: torch.Tensor | None = ...
    token_chunk_offset_ptr: torch.Tensor | None = ...

class GDNAttentionMetadataBuilder(AttentionMetadataBuilder[GDNAttentionMetadata]):
    reorder_batch_threshold: int
    vllm_config: Incomplete
    compilation_config: Incomplete
    speculative_config: Incomplete
    kv_cache_spec: Incomplete
    num_spec: int
    use_spec_decode: bool
    use_full_cuda_graph: bool
    decode_cudagraph_max_bs: int
    spec_state_indices_tensor: torch.Tensor
    non_spec_state_indices_tensor: torch.Tensor
    spec_sequence_masks: torch.Tensor
    spec_token_indx: torch.Tensor
    non_spec_token_indx: torch.Tensor
    spec_query_start_loc: torch.Tensor
    non_spec_query_start_loc: torch.Tensor
    num_accepted_tokens: torch.Tensor
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
        num_accepted_tokens: torch.Tensor | None = None,
        num_decode_draft_tokens_cpu: torch.Tensor | None = None,
        fast_build: bool = False,
    ) -> GDNAttentionMetadata: ...
    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ): ...
