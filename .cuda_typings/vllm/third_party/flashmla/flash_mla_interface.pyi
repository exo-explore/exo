import dataclasses
import torch
from _typeshed import Incomplete

flash_mla_cuda: Incomplete

@dataclasses.dataclass
class FlashMLASchedMeta:
    @dataclasses.dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int
        causal: bool
        is_fp8_kvcache: bool
        topk: int | None
        extra_page_block_size: int | None
        extra_topk: int | None

    have_initialized: bool
    config: Config | None
    tile_scheduler_metadata: torch.Tensor | None
    num_splits: torch.Tensor | None
    def __init__(
        self,
        have_initialized=...,
        config=...,
        tile_scheduler_metadata=...,
        num_splits=...,
    ) -> None: ...
    def __replace__(
        self,
        *,
        have_initialized=...,
        config=...,
        tile_scheduler_metadata=...,
        num_splits=...,
    ) -> None: ...

def get_mla_metadata(*args, **kwargs) -> tuple[FlashMLASchedMeta, None]: ...
def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

class FlashAttnVarlenFunc(torch.autograd.Function):
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_qo: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        max_seqlen_qo: int,
        max_seqlen_kv: int,
        causal: bool = False,
        softmax_scale: float | None = None,
        is_varlen: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def backward(ctx, do: torch.Tensor, dlse: torch.Tensor): ...

def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def flash_attn_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
def flash_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    cu_seqlens_qo: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    max_seqlen_qo: int,
    max_seqlen_kv: int,
    head_dim_qk: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    deterministic: bool = False,
    is_varlen: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]: ...
