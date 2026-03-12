import torch
from _typeshed import Incomplete
from vllm.triton_utils import tl as tl, triton as triton

BT_LIST: Incomplete
USE_DEFAULT_FLA_NORM: Incomplete

@triton.jit
def l2norm_fwd_kernel1(x, y, D, BD: tl.constexpr, eps): ...
def l2norm_fwd_kernel(
    x, y, eps, NB, T, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr
): ...
@triton.jit
def l2norm_fwd_kernel2(
    X, Y, eps, M, N: tl.constexpr, BD: tl.constexpr, MBLOCK: tl.constexpr
): ...
def l2norm_fwd(
    x: torch.Tensor, eps: float = 1e-06, output_dtype: torch.dtype | None = None
): ...
