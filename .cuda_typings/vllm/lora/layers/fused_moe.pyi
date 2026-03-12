import torch
import torch.nn as nn
from .utils import try_get_optimal_moe_lora_config as try_get_optimal_moe_lora_config
from _typeshed import Incomplete
from transformers import PretrainedConfig
from vllm import envs as envs
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.distributed.utils import divide as divide
from vllm.lora.layers.base import BaseLayerWithLoRA as BaseLayerWithLoRA
from vllm.lora.ops.triton_ops.utils import get_lora_op_configs as get_lora_op_configs
from vllm.model_executor.layers.fused_moe import FusedMoE as FusedMoE
from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
    MarlinExperts as MarlinExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts as TritonExperts,
)
from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod as FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
    UnfusedOAITritonExperts as UnfusedOAITritonExperts,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEKernel as FusedMoEKernel,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular as MoEPrepareAndFinalizeNoDPEPModular,
)

class FusedMoEWithLoRA(BaseLayerWithLoRA):
    base_layer: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    device: Incomplete
    def __init__(self, base_layer: FusedMoE) -> None: ...
    max_loras: Incomplete
    fully_sharded: Incomplete
    adapter_enabled: Incomplete
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
    def forward(self, *args, **kwargs): ...
    def maybe_all_reduce_tensor_model_parallel(self, *args, **kwargs): ...
    @property
    def quant_method(self): ...
    @property
    def is_internal_router(self) -> bool: ...
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...

class FusedMoE3DWithLoRA(FusedMoEWithLoRA):
    def __init__(self, base_layer) -> None: ...
    max_loras: Incomplete
    fully_sharded: Incomplete
    adapter_enabled: Incomplete
    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None: ...
    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ): ...
    @property
    def w13_input_size(self): ...
    @property
    def w13_output_size(self): ...
    @property
    def w2_input_size(self): ...
    @property
    def w2_output_size(self): ...
    @classmethod
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...
