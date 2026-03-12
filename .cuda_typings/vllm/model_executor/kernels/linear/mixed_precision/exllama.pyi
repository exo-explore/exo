import torch
from .MPLinearKernel import (
    MPLinearKernel as MPLinearKernel,
    MPLinearLayerConfig as MPLinearLayerConfig,
)
from _typeshed import Incomplete
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter as BasevLLMParameter,
    permute_param_layout_ as permute_param_layout_,
)
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import scalar_types as scalar_types

class ExllamaLinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES: Incomplete
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]: ...
    w_zp_name: str
    w_gidx_name: str
    def process_weights_after_loading(self, layer: torch.nn.Module): ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
