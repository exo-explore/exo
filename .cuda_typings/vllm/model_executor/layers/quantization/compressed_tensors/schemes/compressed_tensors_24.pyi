import torch
from _typeshed import Incomplete
from collections.abc import Callable
from compressed_tensors.quantization import QuantizationArgs
from typing import Any
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)

__all__ = ["CompressedTensors24"]

class CompressedTensors24(CompressedTensorsScheme):
    quantized: Incomplete
    weight_quant: Incomplete
    input_quant: Incomplete
    do_sparse_decompress: Incomplete
    model_compressor: Incomplete
    quant_fp8: Incomplete
    def __init__(
        self,
        quantized: bool = False,
        weight_quant: QuantizationArgs | None = None,
        input_quant: QuantizationArgs | None = None,
        model_compression_config: dict[str, Any] | None = None,
    ) -> None: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    weights_dtype: torch.dtype
    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size: int,
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
