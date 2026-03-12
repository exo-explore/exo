import torch
import torch.nn as nn
from .base_linear import BaseLinearLayerWithLoRA as BaseLinearLayerWithLoRA
from _typeshed import Incomplete
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.model_executor.layers.linear import ReplicatedLinear as ReplicatedLinear

class ReplicatedLinearWithLoRA(BaseLinearLayerWithLoRA):
    output_size: Incomplete
    n_slices: int
    def __init__(self, base_layer: ReplicatedLinear) -> None: ...
    def forward(
        self, input_: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]: ...
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...
    def slice_lora_a(
        self, lora_a: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]: ...
    def slice_lora_b(
        self, lora_b: torch.Tensor | list[torch.Tensor | None]
    ) -> torch.Tensor | list[torch.Tensor | None]: ...
