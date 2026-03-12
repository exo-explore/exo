import torch
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (
    CompressedTensorsLinearTransformMethod,
    TransformTuple,
)

__all__ = ["is_qutlass_fp4_scheme", "QutlassNvFP4LinearMethod"]

def is_qutlass_fp4_scheme(
    quant_scheme: CompressedTensorsScheme | None, input_tfms: dict[int, TransformTuple]
) -> bool: ...

class QutlassNvFP4LinearMethod(CompressedTensorsLinearTransformMethod):
    def create_weights(
        self,
        layer,
        input_size_per_partition,
        output_partition_sizes,
        input_size,
        output_size,
        params_dtype,
        **extra_weight_attrs,
    ): ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
