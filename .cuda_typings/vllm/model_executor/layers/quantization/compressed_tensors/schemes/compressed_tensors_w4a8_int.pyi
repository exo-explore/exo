import torch
from _typeshed import Incomplete
from collections.abc import Callable
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)

__all__ = ["CompressedTensorsW4A8Int"]

class CompressedTensorsW4A8Int(CompressedTensorsScheme):
    strategy: Incomplete
    group_size: Incomplete
    is_static_input_scheme: Incomplete
    input_symmetric: Incomplete
    quant_type: Incomplete
    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        is_static_input_scheme: bool = False,
        input_symmetric: bool = True,
    ) -> None: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    kernel: Incomplete
    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
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
