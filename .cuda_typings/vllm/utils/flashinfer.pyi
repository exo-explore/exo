import functools
import torch
from _typeshed import Incomplete

__all__ = [
    "has_flashinfer",
    "flashinfer_trtllm_fp8_block_scale_moe",
    "flashinfer_cutlass_fused_moe",
    "flashinfer_cutedsl_grouped_gemm_nt_masked",
    "flashinfer_fp4_quantize",
    "silu_and_mul_scaled_nvfp4_experts_quantize",
    "scaled_fp4_grouped_quantize",
    "nvfp4_block_scale_interleave",
    "trtllm_fp4_block_scale_moe",
    "autotune",
    "has_flashinfer_moe",
    "has_flashinfer_comm",
    "has_flashinfer_all2all",
    "has_flashinfer_cutlass_fused_moe",
    "has_flashinfer_cutedsl_grouped_gemm_nt_masked",
    "has_flashinfer_fp8_blockscale_gemm",
    "has_nvidia_artifactory",
    "supports_trtllm_attention",
    "can_use_trtllm_attention",
    "use_trtllm_attention",
    "flashinfer_scaled_fp4_mm",
    "flashinfer_scaled_fp8_mm",
    "flashinfer_quant_nvfp4_8x4_sf_layout",
    "flashinfer_fp8_blockscale_gemm",
    "should_use_flashinfer_for_blockscale_fp8_gemm",
    "is_flashinfer_fp8_blockscale_gemm_supported",
]

@functools.cache
def has_flashinfer() -> bool: ...

flashinfer_trtllm_fp8_block_scale_moe: Incomplete
flashinfer_cutlass_fused_moe: Incomplete
flashinfer_cutedsl_grouped_gemm_nt_masked: Incomplete
flashinfer_fp4_quantize: Incomplete
silu_and_mul_scaled_nvfp4_experts_quantize: Incomplete
scaled_fp4_grouped_quantize: Incomplete
nvfp4_block_scale_interleave: Incomplete
trtllm_fp4_block_scale_moe: Incomplete
autotune: Incomplete

@functools.cache
def has_flashinfer_comm() -> bool: ...
@functools.cache
def has_flashinfer_all2all() -> bool: ...
@functools.cache
def has_flashinfer_moe() -> bool: ...
@functools.cache
def has_flashinfer_cutlass_fused_moe() -> bool: ...
@functools.cache
def has_flashinfer_cutedsl_grouped_gemm_nt_masked() -> bool: ...
@functools.cache
def has_nvidia_artifactory() -> bool: ...
@functools.cache
def supports_trtllm_attention() -> bool: ...
def can_use_trtllm_attention(num_qo_heads: int, num_kv_heads: int) -> bool: ...
def use_trtllm_attention(
    num_qo_heads: int,
    num_kv_heads: int,
    num_tokens: int,
    max_seq_len: int,
    dcp_world_size: int,
    kv_cache_dtype: str,
    q_dtype: torch.dtype,
    is_prefill: bool,
    force_use_trtllm: bool | None = None,
    has_sinks: bool = False,
    has_spec: bool = False,
) -> bool: ...
def flashinfer_scaled_fp4_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    block_scale_a: torch.Tensor,
    block_scale_b: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor: ...
def flashinfer_scaled_fp8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...
def flashinfer_quant_nvfp4_8x4_sf_layout(
    a: torch.Tensor, a_global_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...

flashinfer_fp8_blockscale_gemm: Incomplete

@functools.cache
def has_flashinfer_fp8_blockscale_gemm() -> bool: ...
@functools.cache
def is_flashinfer_fp8_blockscale_gemm_supported() -> bool: ...
def should_use_flashinfer_for_blockscale_fp8_gemm(
    is_flashinfer_supported: bool,
    output_dtype: torch.dtype,
    input: torch.Tensor,
    weight: torch.Tensor,
): ...
