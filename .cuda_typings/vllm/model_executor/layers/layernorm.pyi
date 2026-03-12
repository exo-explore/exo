import torch
import torch.nn as nn
from _typeshed import Incomplete
from vllm import envs as envs
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.layers.batch_invariant import (
    rms_norm_batch_invariant as rms_norm_batch_invariant,
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.platforms import current_platform as current_platform

logger: Incomplete

def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
) -> torch.Tensor: ...
def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def poly_norm(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, variance_epsilon: float
) -> torch.Tensor: ...
def dispatch_rocm_rmsnorm_func(
    with_fused_add: bool, dtype: torch.dtype, use_aiter: bool = False
): ...

class RMSNorm(CustomOp):
    hidden_size: Incomplete
    variance_epsilon: Incomplete
    variance_size_override: Incomplete
    has_weight: Incomplete
    weight: Incomplete
    rocm_norm_func: Incomplete
    rocm_norm_func_with_add: Incomplete
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-06,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None: ...
    @staticmethod
    def forward_static(
        x: torch.Tensor,
        variance_epsilon: float,
        hidden_size: int,
        orig_dtype: torch.dtype,
        weight: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        variance_size_override: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def forward_native(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def forward_cuda(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def forward_hip(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def forward_xpu(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def extra_repr(self) -> str: ...

class GemmaRMSNorm(CustomOp):
    weight: Incomplete
    variance_epsilon: Incomplete
    def __init__(self, hidden_size: int, eps: float = 1e-06) -> None: ...
    def forward_native(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...
    def forward_cuda(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...

class RMSNormGated(CustomOp):
    eps: Incomplete
    activation: Incomplete
    weight: Incomplete
    group_size: Incomplete
    norm_before_gate: Incomplete
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-05,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        activation: str = "swish",
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor: ...
    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor: ...

class LayerNorm(nn.Module):
    dim: Incomplete
    eps: Incomplete
    weight: Incomplete
    bias: Incomplete
    def __init__(self, dim: int, eps: float = 1e-06) -> None: ...
    def forward(self, x: torch.Tensor): ...
