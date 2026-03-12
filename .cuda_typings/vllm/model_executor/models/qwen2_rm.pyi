import abc
import torch
from .interfaces import SupportsLoRA as SupportsLoRA, SupportsPP as SupportsPP
from .interfaces_base import default_pooling_type as default_pooling_type
from .qwen2 import Qwen2Model as Qwen2Model
from .utils import AutoWeightsLoader as AutoWeightsLoader, maybe_prefix as maybe_prefix
from _typeshed import Incomplete
from collections.abc import Iterable
from torch import nn
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.model_executor.layers.pooler import Pooler as Pooler
from vllm.model_executor.layers.pooler.tokwise import (
    pooler_for_token_classify as pooler_for_token_classify,
)
from vllm.sequence import IntermediateTensors as IntermediateTensors

class Qwen2RewardBaseModel(nn.Module, SupportsLoRA, SupportsPP, metaclass=abc.ABCMeta):
    is_pooling_model: bool
    pooler: Pooler
    packed_modules_mapping: Incomplete
    config: Incomplete
    quant_config: Incomplete
    model: Incomplete
    head_dtype: Incomplete
    score: Incomplete
    make_empty_intermediate_tensors: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors: ...
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]: ...

class Qwen2ForRewardModel(Qwen2RewardBaseModel, metaclass=abc.ABCMeta):
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...

class Qwen2ForProcessRewardModel(Qwen2RewardBaseModel, metaclass=abc.ABCMeta):
    pooler: Incomplete
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None: ...
