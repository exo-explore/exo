import torch
import torch.nn as nn
from _typeshed import Incomplete
from vllm.config.model import LogprobsMode as LogprobsMode
from vllm.utils.platform_utils import is_pin_memory_available as is_pin_memory_available
from vllm.v1.outputs import (
    LogprobsTensors as LogprobsTensors,
    SamplerOutput as SamplerOutput,
)
from vllm.v1.sample.metadata import SamplingMetadata as SamplingMetadata
from vllm.v1.sample.ops.bad_words import apply_bad_words as apply_bad_words
from vllm.v1.sample.ops.logprobs import (
    batched_count_greater_than as batched_count_greater_than,
)
from vllm.v1.sample.ops.penalties import apply_all_penalties as apply_all_penalties
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler as TopKTopPSampler

class Sampler(nn.Module):
    topk_topp_sampler: Incomplete
    pin_memory: Incomplete
    logprobs_mode: Incomplete
    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None: ...
    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput: ...
    @staticmethod
    def apply_temperature(
        logits: torch.Tensor, temp: torch.Tensor, all_random: bool
    ) -> torch.Tensor: ...
    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor: ...
    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor: ...
    @staticmethod
    def gather_logprobs(
        logprobs: torch.Tensor, num_logprobs: int, token_ids: torch.Tensor
    ) -> LogprobsTensors: ...
    def apply_logits_processors(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor: ...
    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor: ...
