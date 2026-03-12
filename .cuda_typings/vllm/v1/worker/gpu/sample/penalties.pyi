import numpy as np
import torch
from _typeshed import Incomplete
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.torch_utils import async_tensor_h2d as async_tensor_h2d
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor as UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState as RequestState

class PenaltiesState:
    req_states: Incomplete
    vocab_size: Incomplete
    device: Incomplete
    repetition_penalty: Incomplete
    frequency_penalty: Incomplete
    presence_penalty: Incomplete
    use_penalty: Incomplete
    prompt_bin_mask: Incomplete
    output_bin_counts: Incomplete
    def __init__(self, req_states: RequestState) -> None: ...
    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None: ...
    def apply_staged_writes(self) -> None: ...
    def apply_penalties(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
        num_speculative_tokens: int,
    ) -> None: ...

def apply_penalties(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    token_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    repetition_penalty: torch.Tensor,
    frequency_penalty: torch.Tensor,
    presence_penalty: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    num_speculative_tokens: int,
) -> None: ...
def bincount(
    expanded_idx_mapping: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    prefill_len: torch.Tensor,
    prompt_bin_mask: torch.Tensor,
    output_bin_counts: torch.Tensor,
    max_prefill_len: int,
) -> None: ...
def use_penalty(sampling_params: SamplingParams) -> bool: ...
