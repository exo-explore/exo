import torch
from .MPLinearKernel import (
    MPLinearKernel as MPLinearKernel,
    MPLinearLayerConfig as MPLinearLayerConfig,
)
from _typeshed import Incomplete
from vllm.model_executor.layers.quantization.utils import (
    replace_parameter as replace_parameter,
)
from vllm.platforms import (
    CpuArchEnum as CpuArchEnum,
    current_platform as current_platform,
)
from vllm.scalar_type import scalar_types as scalar_types

class Dynamic4bitLinearKernel(MPLinearKernel):
    SUPPORTED_QUANT_TYPES: Incomplete
    @classmethod
    def get_min_capability(cls) -> int: ...
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]: ...
    def process_weights_after_loading(self, layer: torch.nn.Module): ...
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor: ...
