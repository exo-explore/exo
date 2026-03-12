from _typeshed import Incomplete
from torch import nn as nn
from transformers import PretrainedConfig as PretrainedConfig
from vllm import envs as envs
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.logger import init_logger as init_logger
from vllm.lora.layers import (
    BaseLayerWithLoRA as BaseLayerWithLoRA,
    ColumnParallelLinearWithLoRA as ColumnParallelLinearWithLoRA,
    ColumnParallelLinearWithShardedLoRA as ColumnParallelLinearWithShardedLoRA,
    FusedMoE3DWithLoRA as FusedMoE3DWithLoRA,
    FusedMoEWithLoRA as FusedMoEWithLoRA,
    LogitsProcessorWithLoRA as LogitsProcessorWithLoRA,
    MergedColumnParallelLinearVariableSliceWithLoRA as MergedColumnParallelLinearVariableSliceWithLoRA,
    MergedColumnParallelLinearWithLoRA as MergedColumnParallelLinearWithLoRA,
    MergedColumnParallelLinearWithShardedLoRA as MergedColumnParallelLinearWithShardedLoRA,
    MergedQKVParallelLinearWithLoRA as MergedQKVParallelLinearWithLoRA,
    MergedQKVParallelLinearWithShardedLoRA as MergedQKVParallelLinearWithShardedLoRA,
    QKVParallelLinearWithLoRA as QKVParallelLinearWithLoRA,
    QKVParallelLinearWithShardedLoRA as QKVParallelLinearWithShardedLoRA,
    ReplicatedLinearWithLoRA as ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA as RowParallelLinearWithLoRA,
    RowParallelLinearWithShardedLoRA as RowParallelLinearWithShardedLoRA,
    VocabParallelEmbeddingWithLoRA as VocabParallelEmbeddingWithLoRA,
)
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.linear import LinearBase as LinearBase
from vllm.model_executor.layers.logits_processor import (
    LogitsProcessor as LogitsProcessor,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead as ParallelLMHead,
)
from vllm.model_executor.models.utils import WeightsMapper as WeightsMapper
from vllm.model_executor.utils import (
    get_moe_expert_mapping as get_moe_expert_mapping,
    get_packed_modules_mapping as get_packed_modules_mapping,
)

logger: Incomplete

def get_captured_lora_counts(max_loras: int, specialize: bool) -> list[int]: ...
def get_lora_id(): ...
def is_moe_model(model: nn.Module) -> bool: ...
def from_layer(
    layer: nn.Module,
    max_loras: int,
    lora_config: LoRAConfig,
    packed_modules_list: list,
    model_config: PretrainedConfig | None = None,
) -> nn.Module: ...
def from_layer_logits_processor(
    layer: LogitsProcessor,
    lm_head: ParallelLMHead,
    max_loras: int,
    lora_config: LoRAConfig,
    model_config: PretrainedConfig | None = None,
) -> LogitsProcessorWithLoRA: ...
def replace_submodule(
    model: nn.Module, module_name: str, new_module: nn.Module
) -> nn.Module: ...
def parse_fine_tuned_lora_name(
    name: str, weights_mapper: WeightsMapper | None = None
) -> tuple[str, bool]: ...
def is_base_embedding_weights(name: str) -> bool: ...
def get_supported_lora_modules(model: nn.Module) -> list[str]: ...
def get_adapter_absolute_path(lora_path: str) -> str: ...
def process_packed_modules_mapping(model: nn.Module) -> dict[str, list[str]]: ...
