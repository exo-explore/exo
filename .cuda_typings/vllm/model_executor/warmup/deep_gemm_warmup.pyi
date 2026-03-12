import torch
from tqdm import tqdm
from vllm.distributed.parallel_state import (
    get_dp_group as get_dp_group,
    is_global_first_rank as is_global_first_rank,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    DeepGemmExperts as DeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.deep_gemm_utils import (
    compute_aligned_M as compute_aligned_M,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE as FusedMoE,
    FusedMoEModularMethod as FusedMoEModularMethod,
)
from vllm.model_executor.layers.fused_moe.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts as TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.linear import LinearBase as LinearBase
from vllm.model_executor.layers.quantization.fp8 import (
    Fp8LinearMethod as Fp8LinearMethod,
)
from vllm.tracing import instrument as instrument
from vllm.utils.deep_gemm import (
    fp8_gemm_nt as fp8_gemm_nt,
    get_mk_alignment_for_contiguous_layout as get_mk_alignment_for_contiguous_layout,
    m_grouped_fp8_gemm_nt_contiguous as m_grouped_fp8_gemm_nt_contiguous,
)
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.platform_utils import num_compute_units as num_compute_units

FP8_GEMM_NT_WARMUP_CACHE: set[torch.Size]
GROUPED_FP8_GEMM_NT_CONTIGUOUS_WARMUP_CACHE: set[torch.Size]

def deepgemm_fp8_gemm_nt_warmup(
    model: torch.nn.Module, max_tokens: int, pbar: tqdm | None = None
): ...
def deepgemm_grouped_fp8_gemm_nt_contiguous_warmup(
    model: torch.nn.Module, max_tokens: int, pbar: tqdm | None = None
): ...
def deep_gemm_warmup(model: torch.nn.Module, max_tokens: int): ...
