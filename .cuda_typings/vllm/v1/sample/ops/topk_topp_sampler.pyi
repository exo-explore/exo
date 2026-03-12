import torch
import torch.nn as nn
from _typeshed import Incomplete
from vllm import envs as envs
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config.model import LogprobsMode as LogprobsMode
from vllm.logger import init_logger as init_logger
from vllm.platforms import (
    CpuArchEnum as CpuArchEnum,
    current_platform as current_platform,
)
from vllm.triton_utils import HAS_TRITON as HAS_TRITON
from vllm.v1.sample.ops.topk_topp_triton import (
    apply_top_k_top_p_triton as apply_top_k_top_p_triton,
)

logger: Incomplete

class TopKTopPSampler(nn.Module):
    logprobs_mode: Incomplete
    forward: Incomplete
    aiter_ops: Incomplete
    def __init__(self, logprobs_mode: LogprobsMode = "raw_logprobs") -> None: ...
    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_cpu(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def forward_hip(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
    def aiter_sample(
        self,
        logits: torch.Tensor,
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        generators: dict[int, torch.Generator],
    ) -> torch.Tensor: ...

def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor: ...
def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor: ...
def apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    allow_cpu_sync: bool = False,
) -> torch.Tensor: ...
def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor: ...
def random_sample(
    probs: torch.Tensor, generators: dict[int, torch.Generator]
) -> torch.Tensor: ...
def flashinfer_sample(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    generators: dict[int, torch.Generator],
) -> torch.Tensor: ...
