import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable, Generator
from compressed_tensors.transform import (
    TransformArgs as TransformArgs,
    TransformConfig as TransformConfig,
    TransformScheme as TransformScheme,
)
from vllm.model_executor.layers.linear import (
    LinearMethodBase as LinearMethodBase,
    WEIGHT_LOADER_V2_SUPPORTED as WEIGHT_LOADER_V2_SUPPORTED,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsScheme as CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.module import (
    HadamardTransform as HadamardTransform,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.utils import (
    TransformTuple as TransformTuple,
)

class CompressedTensorsLinearTransformMethod(LinearMethodBase):
    @classmethod
    def from_schemes(
        cls,
        quant_method: LinearMethodBase,
        quant_scheme: CompressedTensorsScheme | None,
        input_tfms: dict[int, TransformTuple],
        output_tfms: dict[int, TransformTuple],
    ) -> CompressedTensorsLinearTransformMethod: ...
    quant_method: Incomplete
    input_tfms: Incomplete
    output_tfms: Incomplete
    input_transform: HadamardTransform | None
    output_transform: HadamardTransform | None
    def __init__(
        self,
        quant_method: LinearMethodBase,
        input_tfms: dict[int, TransformTuple],
        output_tfms: dict[int, TransformTuple],
    ) -> None: ...
    partition_ranges: Incomplete
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
    def process_weights_after_loading(self, layer) -> None: ...
    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...

def get_linear_transform_schemes(
    layer: torch.nn.Module,
    layer_name: str,
    transform_config: TransformConfig | None,
    packed_modules_mapping: dict[str, list[str]],
) -> tuple[dict[int, TransformTuple], dict[int, TransformTuple]]: ...
def get_schemes_args(
    transform_config: TransformConfig | None,
) -> Generator[tuple[str, TransformScheme, TransformArgs]]: ...
def get_layer_partition_names(
    layer_name: str, packed_modules_mapping: dict[str, list[str]]
) -> list[str]: ...
