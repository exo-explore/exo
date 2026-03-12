import torch
from .base import BaseLayerWithLoRA as BaseLayerWithLoRA
from _typeshed import Incomplete
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.distributed.utils import divide as divide
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    LinearBase as LinearBase,
    ReplicatedLinear as ReplicatedLinear,
    RowParallelLinear as RowParallelLinear,
)
from vllm.platforms import current_platform as current_platform

class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):
    base_layer: Incomplete
    input_size: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    device: Incomplete
    output_slices: tuple[int, ...]
    output_size: int
    n_slices: int
    def __init__(self, base_layer: LinearBase) -> None: ...
    lora_config: Incomplete
    lora_a_stacked: Incomplete
    lora_b_stacked: Incomplete
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
    def apply(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    @property
    def weight(self) -> torch.Tensor: ...
    @property
    def bias(self) -> torch.Tensor | None: ...
