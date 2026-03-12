import torch
import torch.nn as nn
from .utils import input_guard as input_guard
from _typeshed import Incomplete
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.math_utils import cdiv as cdiv, next_power_of_2 as next_power_of_2
from vllm.utils.platform_utils import num_compute_units as num_compute_units

def rms_norm_ref(
    x,
    weight,
    bias,
    z=None,
    eps: float = 1e-06,
    group_size=None,
    norm_before_gate: bool = True,
    upcast: bool = True,
): ...
@triton.jit
def layer_norm_fwd_kernel(
    X,
    Y,
    W,
    B,
    Z,
    Mean,
    Rstd,
    stride_x_row,
    stride_y_row,
    stride_z_row,
    M,
    N: tl.constexpr,
    eps,
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    IS_RMS_NORM: tl.constexpr,
    ACTIVATION: tl.constexpr,
): ...
def calc_rows_per_block(M: int, device: torch.device) -> int: ...
def layer_norm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    z: torch.Tensor = None,
    out: torch.Tensor = None,
    group_size: int = None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
): ...
@input_guard
def layernorm_fn(
    x,
    weight,
    bias,
    z=None,
    eps: float = 1e-06,
    group_size=None,
    norm_before_gate: bool = True,
    is_rms_norm: bool = False,
    activation: str = "swish",
): ...
@input_guard
def rmsnorm_fn(
    x,
    weight,
    bias,
    z=None,
    eps: float = 1e-06,
    group_size=None,
    norm_before_gate: bool = True,
    activation: str = "swish",
): ...

class LayerNormGated(nn.Module):
    eps: Incomplete
    weight: Incomplete
    bias: Incomplete
    group_size: Incomplete
    norm_before_gate: Incomplete
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-05,
        group_size: int | None = None,
        norm_before_gate: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x, z=None): ...

class RMSNormGated(nn.Module):
    eps: Incomplete
    activation: Incomplete
    weight: Incomplete
    group_size: Incomplete
    norm_before_gate: Incomplete
    def __init__(
        self,
        hidden_size,
        eps: float = 1e-05,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        activation: str = "swish",
    ) -> None: ...
    def reset_parameters(self) -> None: ...
    def forward(self, x, z=None): ...
