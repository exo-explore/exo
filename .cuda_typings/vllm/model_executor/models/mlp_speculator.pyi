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
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)

SQRT2: Incomplete

class MLPSpeculatorLayerNorm(nn.Module):
    elementwise_scale_and_shift: Incomplete
    weight: Incomplete
    bias: Incomplete
    eps: Incomplete
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-06,
        elementwise_scale_and_shift: bool = True,
    ) -> None: ...
    def forward(self, x): ...

class MLPSpeculator(nn.Module):
    n_predict: Incomplete
    vocab_size: Incomplete
    emb_dim: Incomplete
    inner_dim: Incomplete
    max_speculative_tokens: Incomplete
    tie_weights: Incomplete
    scale_input: Incomplete
    emb: Incomplete
    proj: Incomplete
    head: Incomplete
    ln: Incomplete
    ln0: Incomplete
    state_weight: Incomplete
    emb_weight: Incomplete
    activation: Incomplete
    config: Incomplete
    logits_processor: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...
