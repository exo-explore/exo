import torch
from .index import prepare_chunk_indices as prepare_chunk_indices
from .op import exp as exp
from .utils import (
    FLA_GDN_FIX_BT as FLA_GDN_FIX_BT,
    check_shared_mem as check_shared_mem,
    is_nvidia_hopper as is_nvidia_hopper,
)
from _typeshed import Incomplete
from vllm.triton_utils import tl as tl, triton as triton

BKV_LIST: Incomplete
NUM_WARPS: Incomplete

def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
): ...
def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor: ...
