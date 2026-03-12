import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.logger import init_logger as init_logger
from vllm.model_executor.kernels.linear import (
    init_int8_linear_kernel as init_int8_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme as CompressedTensorsScheme,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter as BasevLLMParameter,
    ChannelQuantScaleParameter as ChannelQuantScaleParameter,
    ModelWeightParameter as ModelWeightParameter,
    PerTensorScaleParameter as PerTensorScaleParameter,
)

logger: Incomplete

class CompressedTensorsW8A8Int8(CompressedTensorsScheme):
    strategy: Incomplete
    is_static_input_scheme: Incomplete
    input_symmetric: Incomplete
    def __init__(
        self, strategy: str, is_static_input_scheme: bool, input_symmetric: bool
    ) -> None: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    kernel: Incomplete
    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ): ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor: ...
