import torch
from _typeshed import Incomplete
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.forward_context import get_forward_context as get_forward_context
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.platforms import current_platform as current_platform
from vllm.utils.deep_gemm import (
    fp8_mqa_logits as fp8_mqa_logits,
    fp8_mqa_logits_torch as fp8_mqa_logits_torch,
    fp8_paged_mqa_logits as fp8_paged_mqa_logits,
    fp8_paged_mqa_logits_torch as fp8_paged_mqa_logits_torch,
    is_deep_gemm_supported as is_deep_gemm_supported,
)
from vllm.utils.torch_utils import (
    direct_register_custom_op as direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata as DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import (
    pack_seq_triton as pack_seq_triton,
    unpack_seq_triton as unpack_seq_triton,
)
from vllm.v1.worker.workspace import (
    current_workspace_manager as current_workspace_manager,
)

logger: Incomplete

def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor: ...
def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: str,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor: ...

class SparseAttnIndexer(CustomOp):
    k_cache: Incomplete
    quant_block_size: Incomplete
    scale_fmt: Incomplete
    topk_tokens: Incomplete
    head_dim: Incomplete
    max_model_len: Incomplete
    max_total_seq_len: Incomplete
    topk_indices_buffer: Incomplete
    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
    ) -> None: ...
    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ): ...
    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ): ...
    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ): ...
