import abc
import torch
from _typeshed import Incomplete
from abc import abstractmethod
from torch.nn.parameter import Parameter
from vllm.distributed import (
    divide as divide,
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
    split_tensor_along_last_dim as split_tensor_along_last_dim,
    tensor_model_parallel_all_gather as tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce as tensor_model_parallel_all_reduce,
)
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import PluggableLayer as PluggableLayer
from vllm.model_executor.layers.batch_invariant import (
    linear_batch_invariant as linear_batch_invariant,
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig as QuantizationConfig,
    QuantizeMethodBase as QuantizeMethodBase,
)
from vllm.model_executor.layers.utils import (
    dispatch_unquantized_gemm as dispatch_unquantized_gemm,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter as BasevLLMParameter,
    BlockQuantScaleParameter as BlockQuantScaleParameter,
    ModelWeightParameter as ModelWeightParameter,
    PackedColumnParameter as PackedColumnParameter,
    PackedvLLMParameter as PackedvLLMParameter,
    PerTensorScaleParameter as PerTensorScaleParameter,
    RowvLLMParameter as RowvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs as set_weight_attrs
from vllm.platforms import current_platform as current_platform

logger: Incomplete
WEIGHT_LOADER_V2_SUPPORTED: Incomplete

def register_weight_loader_v2_supported_method(cls): ...
def adjust_marlin_shard(
    param: Parameter, shard_size: int, shard_offset: int
) -> tuple[int, int]: ...
def adjust_block_scale_shard(
    weight_block_size: tuple[int, ...] | None, shard_size: int, shard_offset: int
) -> tuple[int, int]: ...
def adjust_bitsandbytes_4bit_shard(
    param: Parameter, shard_offsets: dict[str, tuple[int, int]], loaded_shard_id: str
) -> tuple[int, int]: ...
def adjust_scalar_to_fused_array(
    param_data: torch.Tensor, loaded_weight: torch.Tensor, shard_id: int | str
) -> tuple[torch.Tensor, torch.Tensor]: ...

class LinearMethodBase(QuantizeMethodBase, metaclass=abc.ABCMeta):
    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    @abstractmethod
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class UnquantizedLinearMethod(LinearMethodBase):
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ): ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class LinearBase(PluggableLayer):
    input_size: Incomplete
    output_size: Incomplete
    skip_bias_add: Incomplete
    params_dtype: Incomplete
    quant_config: Incomplete
    prefix: Incomplete
    allow_fp8_block_shape_mismatch: bool
    quant_method: QuantizeMethodBase | None
    return_bias: Incomplete
    disable_tp: Incomplete
    tp_rank: Incomplete
    tp_size: Incomplete
    def __init__(
        self,
        input_size: int,
        output_size: int,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None: ...
    def update_param_tp_status(self) -> None: ...

class ReplicatedLinear(LinearBase):
    output_partition_sizes: Incomplete
    bias: Incomplete
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None: ...
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor): ...
    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]: ...
    def extra_repr(self) -> str: ...

class ColumnParallelLinear(LinearBase):
    tp_rank: Incomplete
    tp_size: Incomplete
    input_size_per_partition: Incomplete
    output_size_per_partition: Incomplete
    output_partition_sizes: Incomplete
    gather_output: Incomplete
    bias: Incomplete
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None: ...
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor): ...
    def weight_loader_v2(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ): ...
    def forward(
        self, input_
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]: ...
    def extra_repr(self) -> str: ...

class MergedColumnParallelLinear(ColumnParallelLinear):
    output_sizes: Incomplete
    tp_size: Incomplete
    tp_rank: Incomplete
    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = True,
        gather_output: bool = False,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None: ...
    def validate_shard_id(self, loaded_shard_id: int | tuple[int, ...] | None): ...
    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ): ...
    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: tuple[int, ...] | int | None = None,
    ): ...

class QKVParallelLinear(ColumnParallelLinear):
    hidden_size: Incomplete
    head_size: Incomplete
    v_head_size: Incomplete
    total_num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_heads: Incomplete
    num_kv_heads: int
    num_kv_head_replicas: Incomplete
    output_sizes: Incomplete
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
        v_head_size: int | None = None,
    ) -> None: ...
    def validate_shard_id(self, loaded_shard_id: str | None): ...
    def weight_loader_v2(
        self,
        param: BasevLLMParameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ): ...
    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str | None = None,
    ): ...

class RowParallelLinear(LinearBase):
    tp_rank: Incomplete
    tp_size: Incomplete
    input_size_per_partition: Incomplete
    output_size_per_partition: Incomplete
    output_partition_sizes: Incomplete
    input_is_parallel: Incomplete
    reduce_results: Incomplete
    bias: Incomplete
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: torch.dtype | None = None,
        reduce_results: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        *,
        return_bias: bool = True,
        disable_tp: bool = False,
    ) -> None: ...
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor): ...
    def weight_loader_v2(
        self, param: BasevLLMParameter, loaded_weight: torch.Tensor
    ): ...
    def forward(
        self, input_
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]: ...
    def extra_repr(self) -> str: ...
