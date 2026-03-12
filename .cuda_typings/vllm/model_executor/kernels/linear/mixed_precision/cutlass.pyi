import torch
from .MPLinearKernel import (
    MPLinearKernel as MPLinearKernel,
    MPLinearLayerConfig as MPLinearLayerConfig,
)
from _typeshed import Incomplete
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8 as QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape as GroupShape,
    convert_bf16_scales_to_fp8 as convert_bf16_scales_to_fp8,
    convert_packed_uint4b8_to_signed_int4_inplace as convert_packed_uint4b8_to_signed_int4_inplace,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter as BasevLLMParameter,
    permute_param_layout_ as permute_param_layout_,
)
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import scalar_types as scalar_types

class CutlassW4A8LinearKernel(MPLinearKernel):
    quant_fp8: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]: ...
    def process_weights_after_loading(self, layer: torch.nn.Module): ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
