import torch
from _typeshed import Incomplete
from vllm.triton_utils import tl as tl, triton as triton

AWQ_TRITON_SUPPORTED_GROUP_SIZES: Incomplete

@triton.jit
def awq_dequantize_kernel(
    qweight_ptr,
    scales_ptr,
    zeros_ptr,
    group_size,
    result_ptr,
    num_cols,
    num_rows,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
): ...
@triton.jit
def awq_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    zeros_ptr,
    scales_ptr,
    M,
    N,
    K,
    group_size,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
): ...
def awq_dequantize_triton(
    qweight: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    block_size_x: int = 32,
    block_size_y: int = 32,
) -> torch.Tensor: ...
def awq_gemm_triton(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
) -> torch.Tensor: ...
