import numpy as np
import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils import random_uuid as random_uuid

class InputBuffers:
    max_num_reqs: Incomplete
    max_num_tokens: Incomplete
    device: Incomplete
    input_ids: Incomplete
    positions: Incomplete
    query_start_loc: Incomplete
    seq_lens: Incomplete
    dcp_local_seq_lens: Incomplete
    def __init__(
        self, max_num_reqs: int, max_num_tokens: int, device: torch.device
    ) -> None: ...

@dataclass
class InputBatch:
    req_ids: list[str]
    num_reqs: int
    num_reqs_after_padding: int
    idx_mapping: torch.Tensor
    idx_mapping_np: np.ndarray
    expanded_idx_mapping: torch.Tensor
    expanded_local_pos: torch.Tensor
    num_scheduled_tokens: np.ndarray
    num_tokens: int
    num_tokens_after_padding: int
    num_draft_tokens: int
    query_start_loc: torch.Tensor
    query_start_loc_np: np.ndarray
    seq_lens: torch.Tensor
    dcp_local_seq_lens: torch.Tensor | None
    input_ids: torch.Tensor
    positions: torch.Tensor
    logits_indices: torch.Tensor
    cu_num_logits: torch.Tensor
    cu_num_logits_np: np.ndarray
    has_structured_output_reqs: bool
    @classmethod
    def make_dummy(
        cls, num_reqs: int, num_tokens: int, input_buffers: InputBuffers
    ) -> InputBatch: ...

def prepare_prefill_inputs(
    input_ids: torch.Tensor,
    next_prefill_tokens: torch.Tensor,
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    prefill_len: torch.Tensor,
    num_computed_tokens: torch.Tensor,
) -> None: ...
def prepare_pos_seq_lens(
    idx_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    pos: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None: ...
def combine_sampled_and_draft_tokens(
    input_ids: torch.Tensor,
    idx_mapping: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    prefill_len: torch.Tensor,
    draft_tokens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    num_logits: int,
) -> torch.Tensor: ...
def get_num_sampled_and_rejected(
    num_sampled: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def post_update(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    last_sampled_tokens: torch.Tensor,
    output_bin_counts: torch.Tensor,
    sampled_tokens: torch.Tensor,
    num_sampled: torch.Tensor,
    num_rejected: torch.Tensor,
    query_start_loc: torch.Tensor,
    all_token_ids: torch.Tensor,
    total_len: torch.Tensor,
) -> None: ...
def post_update_pool(
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> None: ...
def expand_idx_mapping(
    idx_mapping: torch.Tensor,
    total_num_logits: int,
    cu_num_logits: torch.Tensor,
    max_expand_len: int,
) -> tuple[torch.Tensor, torch.Tensor]: ...
