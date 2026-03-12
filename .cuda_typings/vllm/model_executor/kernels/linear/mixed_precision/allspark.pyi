import torch
from .MPLinearKernel import (
    MPLinearKernel as MPLinearKernel,
    MPLinearLayerConfig as MPLinearLayerConfig,
)
from _typeshed import Incomplete
from vllm.model_executor.layers.quantization.utils import (
    replace_parameter as replace_parameter,
)
from vllm.model_executor.layers.quantization.utils.allspark_utils import (
    ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD as ALLSPARK_AMPERE_M_CUBLAS_THRESHOLD,
    check_allspark_supported_dtype_shape as check_allspark_supported_dtype_shape,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter as BasevLLMParameter,
    permute_param_layout_ as permute_param_layout_,
)
from vllm.utils.platform_utils import num_compute_units as num_compute_units

class AllSparkLinearKernel(MPLinearKernel):
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]: ...
    gemm_args: Incomplete
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None: ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
