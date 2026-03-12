import torch
import torch.nn as nn
from .base_linear import BaseLinearLayerWithLoRA as BaseLinearLayerWithLoRA
from .utils import _fully_sharded_can_replace, _not_fully_sharded_can_replace
from _typeshed import Incomplete
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config.lora import LoRAConfig as LoRAConfig
from vllm.distributed import (
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
)
from vllm.distributed.utils import divide as divide
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear as ColumnParallelLinear,
    MergedColumnParallelLinear as MergedColumnParallelLinear,
    QKVParallelLinear as QKVParallelLinear,
)
from vllm.platforms import current_platform as current_platform

class ColumnParallelLinearWithLoRA(BaseLinearLayerWithLoRA):
    is_merged_col_linear: Incomplete
    output_size: Incomplete
    n_slices: int
    def __init__(self, base_layer: ColumnParallelLinear) -> None: ...
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

class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    output_slices: Incomplete
    n_slices: Incomplete
    output_ids: Incomplete
    def __init__(
        self, base_layer: MergedColumnParallelLinear | QKVParallelLinear
    ) -> None: ...
    lora_config: Incomplete
    lora_a_stacked: Incomplete
    lora_b_stacked: Incomplete
    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None: ...
    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
    def slice_lora_b(
        self, lora_b: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ): ...
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...

class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    q_proj_total_size: Incomplete
    q_proj_shard_size: Incomplete
    kv_proj_shard_size: Incomplete
    kv_proj_total_size: Incomplete
    n_slices: int
    def __init__(self, base_layer: QKVParallelLinear) -> None: ...
    q_shard_id: Incomplete
    kv_shard_id: Incomplete
    def slice_lora_b(self, lora_b: torch.Tensor) -> torch.Tensor: ...
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...

class MergedQKVParallelLinearWithLoRA(MergedColumnParallelLinearWithLoRA):
    n_slices: Incomplete
    q_proj_shard_size: Incomplete
    kv_proj_shard_size: Incomplete
    q_shard_id: Incomplete
    kv_shard_id: Incomplete
    output_slices: Incomplete
    output_ids: Incomplete
    def __init__(self, base_layer: QKVParallelLinear) -> None: ...
    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None: ...
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...

class ColumnParallelLinearWithShardedLoRA(ColumnParallelLinearWithLoRA):
    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor: ...
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

class MergedColumnParallelLinearWithShardedLoRA(MergedColumnParallelLinearWithLoRA):
    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
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

class QKVParallelLinearWithShardedLoRA(QKVParallelLinearWithLoRA):
    def slice_lora_a(self, lora_a: torch.Tensor) -> torch.Tensor: ...
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

class MergedQKVParallelLinearWithShardedLoRA(MergedQKVParallelLinearWithLoRA):
    def slice_lora_a(
        self, lora_a: list[torch.Tensor | None]
    ) -> list[torch.Tensor | None]: ...
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

class MergedColumnParallelLinearVariableSliceWithLoRA(
    MergedColumnParallelLinearWithLoRA
):
    @classmethod
    @_not_fully_sharded_can_replace
    def can_replace_layer(
        cls,
        source_layer: nn.Module,
        lora_config: LoRAConfig,
        packed_modules_list: list,
        model_config: PretrainedConfig | None = None,
    ) -> bool: ...
    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ): ...
