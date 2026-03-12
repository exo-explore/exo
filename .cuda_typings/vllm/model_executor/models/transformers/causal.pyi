import abc
import torch
from _typeshed import Incomplete
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForTextGeneration as VllmModelForTextGeneration,
)
from vllm.model_executor.models.utils import (
    PPMissingLayer as PPMissingLayer,
    maybe_prefix as maybe_prefix,
)

class CausalMixin(VllmModelForTextGeneration, metaclass=abc.ABCMeta):
    lm_head: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None: ...
