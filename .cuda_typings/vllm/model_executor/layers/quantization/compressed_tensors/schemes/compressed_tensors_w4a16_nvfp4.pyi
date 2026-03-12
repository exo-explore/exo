import torch
from collections.abc import Callable
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)

__all__ = ["CompressedTensorsW4A16Fp4"]

class CompressedTensorsW4A16Fp4(CompressedTensorsScheme):
    group_size: int
    def __init__(self) -> None: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
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
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
