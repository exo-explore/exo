import torch
import torch.nn as nn
from transformers import PretrainedConfig as PretrainedConfig
from typing import overload
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.lora.punica_wrapper import PunicaWrapperBase as PunicaWrapperBase

class BaseLayerWithLoRA(nn.Module):
    @overload
    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
    @overload
    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor: ...
    @overload
    def slice_lora_b(
        self, lora_b: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
    @overload
    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor: ...
    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None: ...
    def reset_lora(self, index: int): ...
    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ): ...
    punica_wrapper: PunicaWrapperBase
    def set_mapping(self, punica_wrapper) -> None: ...
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...
