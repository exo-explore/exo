import numpy as np
import torch
from _typeshed import Incomplete
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.triton_utils import tl as tl, triton as triton
from vllm.v1.worker.gpu.buffer_utils import (
    StagedWriteTensor as StagedWriteTensor,
    UvaBackedTensor as UvaBackedTensor,
)

MAX_NUM_ALLOWED_TOKEN_IDS: int
MAX_NUM_LOGIT_BIAS_TOKENS: int
MAX_NUM_STOP_TOKEN_IDS: int

class LogitBiasState:
    max_num_reqs: Incomplete
    num_allowed_token_ids: Incomplete
    allowed_token_ids: Incomplete
    num_logit_bias: Incomplete
    logit_bias_token_ids: Incomplete
    logit_bias: Incomplete
    min_lens: Incomplete
    num_stop_token_ids: Incomplete
    stop_token_ids: Incomplete
    use_logit_bias: Incomplete
    def __init__(self, max_num_reqs: int, device: torch.device) -> None: ...
    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None: ...
    def apply_staged_writes(self) -> None: ...
    def apply_logit_bias(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
    ) -> None: ...

def apply_logit_bias(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    pos: torch.Tensor,
    num_allowed_token_ids: torch.Tensor,
    allowed_token_ids: torch.Tensor,
    num_logit_bias: torch.Tensor,
    logit_bias_token_ids: torch.Tensor,
    logit_bias: torch.Tensor,
    min_lens: torch.Tensor,
    num_stop_token_ids: torch.Tensor,
    stop_token_ids: torch.Tensor,
) -> None: ...
