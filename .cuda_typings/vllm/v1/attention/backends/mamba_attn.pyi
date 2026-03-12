import abc
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import Any, TypeVar
from vllm.config import VllmConfig as VllmConfig
from vllm.utils.math_utils import cdiv as cdiv
from vllm.v1.attention.backend import (
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

M = TypeVar("M", bound="BaseMambaAttentionMetadata")

@dataclass
class BaseMambaAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_reqs: int
    has_initial_states_p: torch.Tensor | None
    query_start_loc_p: torch.Tensor | None
    num_computed_tokens_p: torch.Tensor | None
    state_indices_tensor_p: torch.Tensor | None
    state_indices_tensor_d: torch.Tensor | None
    query_start_loc_d: torch.Tensor | None
    num_accepted_tokens: torch.Tensor | None
    block_idx_last_scheduled_token: torch.Tensor | None
    block_idx_first_scheduled_token_p: torch.Tensor | None
    block_idx_last_computed_token: torch.Tensor | None
    seq_lens: torch.Tensor
    cu_chunk_seqlen_p: torch.Tensor | None = ...
    last_chunk_indices_p: torch.Tensor | None = ...
    nums_dict: dict | None = ...
    batch_ptr: torch.Tensor | None = ...
    token_chunk_offset_ptr: torch.Tensor | None = ...

class BaseMambaAttentionMetadataBuilder(AttentionMetadataBuilder[M], abc.ABC):
    metadata_cls: type[M]
    reorder_batch_threshold: int
    supports_update_block_table: bool
    speculative_config: Incomplete
    compilation_config: Incomplete
    num_spec_tokens: int
    use_spec_decode: Incomplete
    decode_cudagraph_max_bs: int
    state_indices_tensor_d: torch.Tensor
    block_idx_last_scheduled_token: torch.Tensor
    block_idx_last_computed_token: torch.Tensor
    decode_num_accepted_tokens: torch.Tensor
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...
    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> M: ...
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        *,
        num_accepted_tokens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> M: ...
    def update_block_table(
        self, metadata: M, blk_table: torch.Tensor, slot_mapping: torch.Tensor
    ) -> M: ...
