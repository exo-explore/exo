import torch
from _typeshed import Incomplete
from collections.abc import Callable
from compressed_tensors.quantization import ActivationOrdering
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)

__all__ = ["CompressedTensorsWNA16"]

class CompressedTensorsWNA16(CompressedTensorsScheme):
    pack_factor: Incomplete
    strategy: Incomplete
    symmetric: Incomplete
    group_size: Incomplete
    has_g_idx: Incomplete
    layer_name: Incomplete
    quant_type: Incomplete
    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        symmetric: bool | None = True,
        actorder: ActivationOrdering | None = None,
        layer_name: str | None = None,
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
