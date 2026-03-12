import numpy as np
import torch
import torch.nn as nn
from _typeshed import Incomplete
from collections.abc import Iterable
from transformers import ParakeetFeatureExtractor, PretrainedConfig as PretrainedConfig
from vllm.model_executor.layers.activation import (
    ReLUSquaredActivation as ReLUSquaredActivation,
)
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader as default_weight_loader,
)
from vllm.transformers_utils.configs.parakeet import (
    ExtractorConfig as ExtractorConfig,
    ParakeetConfig as ParakeetConfig,
)

class ParakeetProjection(nn.Module):
    norm: Incomplete
    linear1: Incomplete
    activation: Incomplete
    linear2: Incomplete
    def __init__(self, config: ParakeetConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor: ...

class ProjectedParakeet(nn.Module):
    config: Incomplete
    encoder: Incomplete
    projection: Incomplete
    def __init__(
        self,
        config: PretrainedConfig,
        *,
        dtype: torch.dtype,
        llm_hidden_size: int,
        max_model_len: int,
    ) -> None: ...
    def forward(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class ParakeetExtractor(ParakeetFeatureExtractor):
    config: Incomplete
    def __init__(self, config: PretrainedConfig) -> None: ...
    def audio_token_count(self, audio_len: int) -> int: ...
    def __call__(self, raw_speech: list[np.ndarray], *args, **kwargs): ...
    def audio_length(self, audio_tokens: int) -> int: ...
