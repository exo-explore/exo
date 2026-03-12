import torch
import torch.nn as nn
from .utils import maybe_prefix as maybe_prefix
from _typeshed import Incomplete
from collections.abc import Iterable
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)

class ResidualBlock(nn.Module):
    layers: Incomplete
    act: Incomplete
    def __init__(
        self, config: VllmConfig, hidden_size: int, num_layers: int
    ) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class Medusa(nn.Module):
    config: Incomplete
    blocks: Incomplete
    orig_vocab_size: Incomplete
    truncated_vocab_size: Incomplete
    lm_head: Incomplete
    lm_heads: Incomplete
    logits_processor: Incomplete
    token_map: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> list[torch.Tensor]: ...
    def compute_logits(
        self, hidden_states: list[torch.Tensor]
    ) -> list[torch.Tensor]: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
