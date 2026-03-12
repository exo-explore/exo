import torch
import torch.nn as nn
from .base_linear import BaseLinearLayerWithLoRA as BaseLinearLayerWithLoRA
from .utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace
from _typeshed import Incomplete
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.distributed import (
    split_tensor_along_last_dim as split_tensor_along_last_dim,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import RowParallelLinear as RowParallelLinear
from vllm.platforms import current_platform as current_platform

class RowParallelLinearWithLoRA(BaseLinearLayerWithLoRA):
    input_size: Incomplete
    output_size: Incomplete
    n_slices: int
    def __init__(self, base_layer: RowParallelLinear) -> None: ...
    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor: ...
    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor: ...
    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]: ...
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...

class RowParallelLinearWithShardedLoRA(RowParallelLinearWithLoRA):
    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor: ...
    def apply(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @classmethod
    @_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...
