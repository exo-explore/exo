import torch
from _typeshed import Incomplete
from typing import Callable
from vllm.vllm_flash_attn.cute import cute_dsl_ptxas as cute_dsl_ptxas, utils as utils
from vllm.vllm_flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch as BlockSparseTensorsTorch,
    normalize_block_sparse_config as normalize_block_sparse_config,
    normalize_block_sparse_config_bwd as normalize_block_sparse_config_bwd,
    to_cute_block_sparse_tensors as to_cute_block_sparse_tensors,
)
from vllm.vllm_flash_attn.cute.cute_dsl_utils import to_cute_tensor as to_cute_tensor
from vllm.vllm_flash_attn.cute.flash_bwd import (
    FlashAttentionBackwardSm80 as FlashAttentionBackwardSm80,
)
from vllm.vllm_flash_attn.cute.flash_bwd_postprocess import (
    FlashAttentionBackwardPostprocess as FlashAttentionBackwardPostprocess,
)
from vllm.vllm_flash_attn.cute.flash_bwd_preprocess import (
    FlashAttentionBackwardPreprocess as FlashAttentionBackwardPreprocess,
)
from vllm.vllm_flash_attn.cute.flash_bwd_sm100 import (
    FlashAttentionBackwardSm100 as FlashAttentionBackwardSm100,
)
from vllm.vllm_flash_attn.cute.flash_bwd_sm90 import (
    FlashAttentionBackwardSm90 as FlashAttentionBackwardSm90,
)
from vllm.vllm_flash_attn.cute.flash_fwd import (
    FlashAttentionForwardSm90 as FlashAttentionForwardSm90,
)
from vllm.vllm_flash_attn.cute.flash_fwd_combine import (
    FlashAttentionForwardCombine as FlashAttentionForwardCombine,
)
from vllm.vllm_flash_attn.cute.flash_fwd_sm100 import (
    FlashAttentionForwardSm100 as FlashAttentionForwardSm100,
)

def maybe_contiguous(x): ...

torch2cute_dtype_map: Incomplete

def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits): ...

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int | None, int | None] = (None, None),
        learnable_sink: torch.Tensor | None = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: bool | None = None,
        deterministic: bool = False,
        mask_mod: Callable | None = None,
        full_block_cnt: torch.Tensor | None = None,
        full_block_idx: torch.Tensor | None = None,
        mask_block_cnt: torch.Tensor | None = None,
        mask_block_idx: torch.Tensor | None = None,
        block_size: tuple[int, int] | None = None,
    ): ...
    @staticmethod
    def backward(ctx, dout, *args): ...

class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor | None,
        cu_seqlens_k: torch.Tensor | None,
        seqused_q: torch.Tensor | None = None,
        seqused_k: torch.Tensor | None = None,
        max_seqlen_q: int | None = None,
        max_seqlen_k: int | None = None,
        page_table: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        window_size: tuple[int | None, int | None] = (None, None),
        learnable_sink: torch.Tensor | None = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: bool | None = None,
        deterministic: bool = False,
        score_mod: Callable | None = None,
        aux_tensors: list | None = None,
    ): ...
    @staticmethod
    def backward(ctx, dout, *args): ...

def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    learnable_sink: torch.Tensor | None = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    mask_mod: Callable | None = None,
    full_block_cnt: torch.Tensor | None = None,
    full_block_idx: torch.Tensor | None = None,
    mask_block_cnt: torch.Tensor | None = None,
    mask_block_idx: torch.Tensor | None = None,
    block_size: tuple[int, int] | None = None,
): ...
def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int | None, int | None] = (None, None),
    learnable_sink: torch.Tensor | None = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    deterministic: bool = False,
    score_mod: Callable | None = None,
    aux_tensors: list | None = None,
): ...
def flash_attn_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    cu_seqlens: torch.Tensor | None = None,
    seqused: torch.Tensor | None = None,
    return_lse: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]: ...
