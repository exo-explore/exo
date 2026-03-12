import numpy as np
import torch
from _typeshed import Incomplete
from vllm.config.model import LogprobsMode as LogprobsMode
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.v1.worker.gpu.metrics.logits import get_num_nans as get_num_nans
from vllm.v1.worker.gpu.sample.bad_words import BadWordsState as BadWordsState
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample as gumbel_sample
from vllm.v1.worker.gpu.sample.logit_bias import LogitBiasState as LogitBiasState
from vllm.v1.worker.gpu.sample.logprob import (
    compute_topk_logprobs as compute_topk_logprobs,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput as SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState as PenaltiesState
from vllm.v1.worker.gpu.sample.states import (
    NO_LOGPROBS as NO_LOGPROBS,
    SamplingStates as SamplingStates,
)
from vllm.v1.worker.gpu.states import RequestState as RequestState

class Sampler:
    logprobs_mode: Incomplete
    compute_nans: Incomplete
    sampling_states: Incomplete
    penalties_state: Incomplete
    logit_bias_state: Incomplete
    bad_words_state: Incomplete
    num_speculative_tokens: Incomplete
    def __init__(
        self,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        req_states: RequestState,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        num_speculative_tokens: int = 1,
    ) -> None: ...
    def add_request(
        self, req_idx: int, prompt_len: int, sampling_params: SamplingParams
    ) -> None: ...
    def apply_staged_writes(self) -> None: ...
    def __call__(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        cu_num_logits_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> SamplerOutput: ...
    def sample(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        pos: torch.Tensor,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
