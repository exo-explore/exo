import torch
import torch.nn as nn
from .base import BaseLayerWithLoRA as BaseLayerWithLoRA
from _typeshed import Incomplete
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding as VocabParallelEmbedding,
)
from vllm.platforms import current_platform as current_platform

class LogitsProcessorWithLoRA(BaseLayerWithLoRA):
    base_layer: Incomplete
    hidden_size: Incomplete
    dtype: Incomplete
    device: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    sharded_to_full_mapping: Incomplete
    def __init__(
        self,
        base_layer: LogitsProcessor,
        hidden_size: int,
        dtype: torch.dtype,
        device: torch.device,
        sharded_to_full_mapping: list[int] | None,
    ) -> None: ...
    @property
    def logits_as_input(self): ...
    @property
    def vocab_size(self): ...
    @property
    def scale(self): ...
    @property
    def soft_cap(self): ...
    @property
    def use_all_gather(self): ...
    @property
    def org_vocab_size(self): ...
    @property
    def include_gpu_probs_tensor(self): ...
    @property
    def should_modify_greedy_probs_inplace(self): ...
    lora_a_stacked: Incomplete
    lora_b_stacked: Incomplete
    sharded_to_full_mapping_gpu: Incomplete
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
    def forward(self, *args, **kwargs): ...
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...
