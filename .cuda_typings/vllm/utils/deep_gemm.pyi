import functools
import torch
from enum import Enum

__all__ = [
    "calc_diff",
    "DeepGemmQuantScaleFMT",
    "fp8_gemm_nt",
    "m_grouped_fp8_gemm_nt_contiguous",
    "fp8_m_grouped_gemm_nt_masked",
    "fp8_mqa_logits",
    "fp8_mqa_logits_torch",
    "fp8_paged_mqa_logits",
    "fp8_paged_mqa_logits_torch",
    "get_paged_mqa_logits_metadata",
    "per_block_cast_to_fp8",
    "is_deep_gemm_e8m0_used",
    "is_deep_gemm_supported",
    "get_num_sms",
    "should_use_deepgemm_for_fp8_linear",
    "get_col_major_tma_aligned_tensor",
    "get_mk_alignment_for_contiguous_layout",
]

class DeepGemmQuantScaleFMT(Enum):
    FLOAT32 = 0
    FLOAT32_CEIL_UE8M0 = 1
    UE8M0 = 2
    @classmethod
    def init_oracle_cache(cls) -> None: ...
    @classmethod
    def from_oracle(cls) -> DeepGemmQuantScaleFMT: ...

@functools.cache
def is_deep_gemm_supported() -> bool: ...
@functools.cache
def is_deep_gemm_e8m0_used() -> bool: ...
def get_num_sms() -> int: ...
@functools.cache
def get_mk_alignment_for_contiguous_layout() -> list[int]: ...
def get_col_major_tma_aligned_tensor(x: torch.Tensor) -> torch.Tensor: ...
def fp8_gemm_nt(*args, **kwargs): ...
def m_grouped_fp8_gemm_nt_contiguous(*args, **kwargs): ...
def fp8_m_grouped_gemm_nt_masked(*args, **kwargs): ...
def fp8_mqa_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor: ...
def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_size: int, num_sms: int
) -> torch.Tensor: ...
def fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor: ...
def per_block_cast_to_fp8(
    x: torch.Tensor, block_size: list[int] = ..., use_ue8m0: bool = False
) -> tuple[torch.Tensor, torch.Tensor]: ...
def calc_diff(x: torch.Tensor, y: torch.Tensor): ...
def should_use_deepgemm_for_fp8_linear(
    output_dtype: torch.dtype,
    weight: torch.Tensor,
    supports_deep_gemm: bool | None = None,
): ...
def fp8_mqa_logits_torch(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor: ...
def fp8_paged_mqa_logits_torch(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor: ...
