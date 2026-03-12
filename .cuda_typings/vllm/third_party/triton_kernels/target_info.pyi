import triton
from triton.language.target_info import (
    cuda_capability_geq as cuda_capability_geq,
    is_cuda as is_cuda,
    is_hip as is_hip,
    is_hip_cdna3 as is_hip_cdna3,
    is_hip_cdna4 as is_hip_cdna4,
)

__all__ = [
    "cuda_capability_geq",
    "get_cdna_version",
    "has_tma_gather",
    "has_native_mxfp",
    "is_cuda",
    "is_hip",
    "is_hip_cdna3",
    "is_hip_cdna4",
    "num_sms",
]

@triton.constexpr_function
def get_cdna_version(): ...
@triton.constexpr_function
def has_tma_gather(): ...
@triton.constexpr_function
def has_native_mxfp(): ...
def num_sms(): ...
