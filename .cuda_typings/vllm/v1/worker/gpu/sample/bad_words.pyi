import numpy as np
import torch
from _typeshed import Incomplete
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.triton_utils import tl as tl, triton as triton
from vllm.v1.worker.gpu.buffer_utils import (
    StagedWriteTensor as StagedWriteTensor,
    UvaBackedTensor as UvaBackedTensor,
)
from vllm.v1.worker.gpu.states import RequestState as RequestState

MAX_BAD_WORDS_TOTAL_TOKENS: int
MAX_NUM_BAD_WORDS: int

class BadWordsState:
    req_states: Incomplete
    max_num_reqs: Incomplete
    device: Incomplete
    bad_word_token_ids: Incomplete
    bad_word_offsets: Incomplete
    num_bad_words: Incomplete
    def __init__(self, req_states: RequestState) -> None: ...
    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None: ...
    def apply_staged_writes(self) -> None: ...
    def apply_bad_words(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None: ...

def apply_bad_words(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    bad_word_token_ids: torch.Tensor,
    bad_word_offsets: torch.Tensor,
    num_bad_words: torch.Tensor,
    all_token_ids: torch.Tensor,
    prompt_len: torch.Tensor,
    total_len: torch.Tensor,
    input_ids: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    max_num_bad_words: int,
) -> None: ...
