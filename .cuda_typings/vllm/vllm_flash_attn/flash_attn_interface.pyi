import torch
from _typeshed import Incomplete

FA2_UNAVAILABLE_REASON: Incomplete
FA2_AVAILABLE: bool
FA3_UNAVAILABLE_REASON: Incomplete
FA3_AVAILABLE: bool
FA4_UNAVAILABLE_REASON: Incomplete
FA4_AVAILABLE: bool
DEFAULT_FA_VERSION: int

def is_fa_version_supported(fa_version: int) -> bool: ...
def fa_version_unsupported_reason(fa_version: int) -> str | None: ...
def maybe_contiguous(x): ...
def get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=...,
    headdim_v=None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k_new: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    page_size: int | None = None,
    max_seqlen_k_new: int = 0,
    causal: bool = False,
    window_size=(-1, -1),
    has_softcap: bool = False,
    num_splits: int = 0,
    pack_gqa=None,
    sm_margin: int = 0,
): ...
def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal: bool = False,
    window_size: list[int] | None = None,
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table=None,
    return_softmax_lse: bool = False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits: int = 0,
    fa_version: int = ...,
    s_aux=None,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_tot_seqused_k=None,
): ...
def sparse_attn_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal: bool = False,
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    return_softmax_lse: bool = False,
    out=None,
): ...
def sparse_attn_varlen_func(
    q,
    k,
    v,
    block_count,
    block_offset,
    column_count,
    column_index,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p: float = 0.0,
    softmax_scale=None,
    causal: bool = False,
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    *,
    return_softmax_lse: bool = False,
    out=None,
): ...
