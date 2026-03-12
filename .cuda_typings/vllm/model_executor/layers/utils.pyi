import torch
from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm import envs as envs
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.logger import init_logger as init_logger
from vllm.platforms import (
    CpuArchEnum as CpuArchEnum,
    current_platform as current_platform,
)
from vllm.utils.platform_utils import num_compute_units as num_compute_units
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)

logger: Incomplete
MOE_LAYER_ROUTER_GATE_SUFFIXES: Incomplete

def is_layer_moe_router_gate(prefix: str) -> bool: ...
def get_token_bin_counts_and_mask(
    tokens: torch.Tensor, vocab_size: int, num_seqs: int
) -> tuple[torch.Tensor, torch.Tensor]: ...
def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor: ...
def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
): ...
def use_aiter_triton_gemm(n, m, k, dtype): ...
def rocm_unquantized_gemm_impl(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor: ...
def rocm_unquantized_gemm_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor: ...
def rocm_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor: ...
def check_cpu_sgl_kernel(n: int, k: int, dtype: torch.dtype) -> bool: ...
def dispatch_cpu_unquantized_gemm(
    layer: torch.nn.Module, remove_weight: bool
) -> None: ...
def cpu_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
): ...
def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]: ...
