import torch
import torch.nn as nn
from .base import BaseLayerWithLoRA as BaseLayerWithLoRA
from _typeshed import Incomplete
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.platforms import current_platform as current_platform

class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    base_layer: Incomplete
    embeddings_slice: tuple[int, int] | None
    embeddings_weights: torch.Tensor | None
    def __init__(self, base_layer: VocabParallelEmbedding) -> None: ...
    lora_a_stacked: Incomplete
    lora_b_stacked: Incomplete
    lora_a_stacked_2d: Incomplete
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
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...
    @property
    def weight(self): ...
