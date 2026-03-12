import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.triton_utils import tl as tl, triton as triton
from vllm.utils.platform_utils import num_compute_units as num_compute_units
from vllm.utils.torch_utils import is_torch_equal_or_newer as is_torch_equal_or_newer
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)

logger: Incomplete

def matmul_kernel_persistent(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
): ...
def matmul_persistent(
    a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
): ...
@triton.jit
def bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    A_LARGE: tl.constexpr,
    B_LARGE: tl.constexpr,
    C_LARGE: tl.constexpr,
): ...
def log_softmax(input: torch.Tensor, dim: int = -1) -> torch.Tensor: ...
@triton.jit
def mean_kernel(
    input_ptr,
    output_ptr,
    input_stride0,
    input_stride1,
    input_stride2,
    output_stride0,
    output_stride1,
    M,
    N,
    K,
    BLOCK_SIZE: tl.constexpr,
): ...
def mean_dim(
    input: torch.Tensor,
    dim: int,
    keepdim: bool = False,
    dtype: torch.dtype | None = None,
) -> torch.Tensor: ...
def mm_batch_invariant(a, b): ...
def matmul_batch_invariant(a, b, *, out=None): ...
def bmm_batch_invariant(a, b, *, out=None): ...
def addmm_batch_invariant(bias, a, b): ...
def softmax_batch_invariant(input, dim, dtype=None): ...
def mean_batch_invariant(
    input, dim, keepdim: bool = False, dtype: torch.dtype | None = None
): ...
def rms_norm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-06
) -> torch.Tensor: ...
def rms_norm_batch_invariant(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-06
) -> torch.Tensor: ...
def linear_batch_invariant(input, weight, bias=None): ...
def enable_batch_invariant_mode() -> None: ...

VLLM_BATCH_INVARIANT: bool

def vllm_is_batch_invariant() -> bool: ...
def override_envs_for_invariance(attention_backend: AttentionBackendEnum | None): ...
def init_batch_invariance(attention_backend: AttentionBackendEnum | None): ...
