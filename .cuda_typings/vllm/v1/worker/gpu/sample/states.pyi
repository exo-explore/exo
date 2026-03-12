import numpy as np
import torch
from _typeshed import Incomplete
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p as apply_top_k_top_p
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor as UvaBackedTensor
from vllm.v1.worker.gpu.sample.gumbel import apply_temperature as apply_temperature
from vllm.v1.worker.gpu.sample.min_p import apply_min_p as apply_min_p

NO_LOGPROBS: int

class SamplingStates:
    max_num_reqs: Incomplete
    vocab_size: Incomplete
    temperature: Incomplete
    top_k: Incomplete
    top_p: Incomplete
    min_p: Incomplete
    seeds: Incomplete
    num_logprobs: Incomplete
    def __init__(self, max_num_reqs: int, vocab_size: int) -> None: ...
    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None: ...
    def apply_staged_writes(self) -> None: ...
    def apply_temperature(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None: ...
    def apply_min_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None: ...
    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor: ...
    def max_num_logprobs(self, idx_mapping_np: np.ndarray) -> int: ...
