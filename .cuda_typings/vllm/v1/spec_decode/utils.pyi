import torch
from vllm.config import VllmConfig as VllmConfig, replace as replace
from vllm.triton_utils import tl as tl, triton as triton
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata as CommonAttentionMetadata,
)

PADDING_SLOT_ID: int

@triton.jit
def eagle_step_slot_mapping_metadata_kernel(
    positions_ptr,
    block_table_ptr,
    block_table_stride,
    seq_lens_ptr,
    out_clamped_positions_ptr,
    out_slot_mapping_ptr,
    block_size: tl.constexpr,
    max_model_len: tl.constexpr,
    n_blocks_per_req: tl.constexpr,
    PAD_ID: tl.constexpr,
    batch_size,
): ...
def eagle_step_update_slot_mapping_and_metadata(
    positions_1d: torch.Tensor,
    block_table_tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    max_model_len: int,
    out_clamped_positions: torch.Tensor,
    out_slot_mapping: torch.Tensor,
    input_batch_size: int | None = None,
) -> None: ...
@triton.jit
def eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,
    valid_sampled_tokens_count_ptr,
    query_start_loc_gpu_ptr,
    token_indices_to_sample_ptr,
    num_rejected_tokens_gpu_ptr,
    num_reqs,
) -> None: ...
@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,
    discard_request_mask_ptr,
    backup_next_token_ids_ptr,
    next_token_ids_ptr,
    valid_sampled_tokens_count_ptr,
    vocab_size,
    num_sampled_tokens_per_req,
    num_reqs,
    stride_sampled_token_ids,
    BLOCK_SIZE_TOKENS: tl.constexpr,
): ...
def compute_new_slot_mapping(
    cad: CommonAttentionMetadata,
    new_positions: torch.Tensor,
    is_rejected_token_mask: torch.Tensor,
    block_size: int,
    num_new_tokens: int,
    max_model_len: int,
): ...
def create_vllm_config_for_draft_model(
    target_model_vllm_config: VllmConfig,
) -> VllmConfig: ...
def extend_all_queries_by_N(
    common_attn_metadata: CommonAttentionMetadata,
    N: int,
    arange: torch.Tensor,
    new_slot_mapping: torch.Tensor,
) -> CommonAttentionMetadata: ...
@triton.jit
def copy_and_expand_eagle_inputs_kernel(
    target_token_ids_ptr,
    target_positions_ptr,
    next_token_ids_ptr,
    out_input_ids_ptr,
    out_positions_ptr,
    out_is_rejected_token_mask_ptr,
    out_is_masked_token_mask_ptr,
    out_new_token_indices_ptr,
    out_hidden_state_mapping_ptr,
    query_start_loc_ptr,
    query_end_loc_ptr,
    padding_token_id,
    parallel_drafting_token_id,
    total_input_tokens,
    num_padding_slots_per_request,
    shift_input_ids,
    BLOCK_SIZE_TOKENS: tl.constexpr,
): ...
