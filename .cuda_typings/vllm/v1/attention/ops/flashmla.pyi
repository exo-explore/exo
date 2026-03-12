import torch
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.platforms import current_platform as current_platform
from vllm.third_party.flashmla.flash_mla_interface import (
    FlashMLASchedMeta as FlashMLASchedMeta,
    flash_attn_varlen_func as flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func as flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func as flash_attn_varlen_qkvpacked_func,
    flash_mla_sparse_fwd as flash_mla_sparse_fwd,
    flash_mla_with_kvcache as flash_mla_with_kvcache,
    get_mla_metadata as get_mla_metadata,
)

logger: Incomplete

def is_flashmla_dense_supported() -> tuple[bool, str | None]: ...
def is_flashmla_sparse_supported() -> tuple[bool, str | None]: ...

class FlashMLASchedMeta: ...

def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor, num_q_tokens_per_head_k: int, num_heads_k: int
) -> tuple[torch.Tensor, torch.Tensor]: ...
def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]: ...
