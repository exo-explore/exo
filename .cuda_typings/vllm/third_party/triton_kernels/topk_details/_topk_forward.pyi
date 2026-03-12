import triton
import triton.language as tl

@triton.jit
def get_topmask_and_fullmask(x): ...
@triton.jit
def fpval_to_key(x): ...
@triton.jit
def key_to_fpval(x): ...
@triton.jit
def indx_to_key(indx, N_EXPTS_PAD: tl.constexpr): ...
@triton.jit
def key_to_indx(indx, N_EXPTS_PAD: tl.constexpr): ...
@triton.jit
def streaming_topk(
    X,
    stride_xm,
    n_expts_tot,
    offs_m,
    mask_m,
    N_EXPTS_PAD: tl.constexpr,
    N_EXPTS_ACT: tl.constexpr,
    BLOCK_N: tl.constexpr,
): ...
