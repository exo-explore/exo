import torch
from _typeshed import Incomplete
from collections.abc import Callable
from compressed_tensors.quantization import QuantizationArgs
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)

__all__ = ["CompressedTensorsW8A16Fp8"]

class CompressedTensorsW8A16Fp8(CompressedTensorsScheme):
    weight_quant: Incomplete
    strategy: Incomplete
    is_static_input_scheme: Incomplete
    weight_block_size: Incomplete
    def __init__(
        self, weight_quant: QuantizationArgs, is_static_input_scheme: bool
    ) -> None: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ): ...
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
