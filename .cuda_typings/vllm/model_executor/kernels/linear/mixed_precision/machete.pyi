import torch
from .MPLinearKernel import (
    MPLinearKernel as MPLinearKernel,
    MPLinearLayerConfig as MPLinearLayerConfig,
)
from _typeshed import Incomplete
from vllm.model_executor.layers.quantization.utils.machete_utils import (
    check_machete_supports_shape as check_machete_supports_shape,
    query_machete_supported_group_sizes as query_machete_supported_group_sizes,
    query_machete_supported_quant_types as query_machete_supported_quant_types,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32 as pack_quantized_values_into_int32,
    unpack_quantized_values_into_int32 as unpack_quantized_values_into_int32,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter as BasevLLMParameter,
    permute_param_layout_ as permute_param_layout_,
)
from vllm.platforms import current_platform as current_platform

class MacheteLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]: ...
    act_perm: Incomplete
    def process_weights_after_loading(self, layer: torch.nn.Module): ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
