import torch
from .MPLinearKernel import (
    MPLinearKernel as MPLinearKernel,
    MPLinearLayerConfig as MPLinearLayerConfig,
)
from _typeshed import Incomplete
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    MARLIN_SUPPORTED_GROUP_SIZES as MARLIN_SUPPORTED_GROUP_SIZES,
    apply_gptq_marlin_linear as apply_gptq_marlin_linear,
    check_marlin_supports_shape as check_marlin_supports_shape,
    marlin_act_int8_process_scales as marlin_act_int8_process_scales,
    marlin_is_k_full as marlin_is_k_full,
    marlin_make_empty_g_idx as marlin_make_empty_g_idx,
    marlin_make_workspace_new as marlin_make_workspace_new,
    marlin_permute_bias as marlin_permute_bias,
    marlin_permute_scales as marlin_permute_scales,
    marlin_sort_g_idx as marlin_sort_g_idx,
    marlin_zero_points as marlin_zero_points,
    query_marlin_supported_quant_types as query_marlin_supported_quant_types,
    unpack_cols as unpack_cols,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter as BasevLLMParameter,
    permute_param_layout_ as permute_param_layout_,
)
from vllm.platforms import current_platform as current_platform
from vllm.scalar_type import scalar_types as scalar_types

class MarlinLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]: ...
    is_k_full: Incomplete
    workspace: Incomplete
    w_gidx_name: str
    w_zp_name: str
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
