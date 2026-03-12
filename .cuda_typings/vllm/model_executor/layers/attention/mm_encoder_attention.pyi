import numpy as np
import torch
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.model_executor.custom_op import CustomOp as CustomOp
from vllm.model_executor.models.vision import (
    get_vit_attn_backend as get_vit_attn_backend,
)
from vllm.utils.math_utils import round_up as round_up
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version as get_flash_attn_version,
)
from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum as AttentionBackendEnum,
)
from vllm.v1.attention.ops.vit_attn_wrappers import (
    vit_flash_attn_wrapper as vit_flash_attn_wrapper,
    vit_flashinfer_wrapper as vit_flashinfer_wrapper,
    vit_torch_sdpa_wrapper as vit_torch_sdpa_wrapper,
    vit_triton_attn_wrapper as vit_triton_attn_wrapper,
)

logger: Incomplete
FLASHINFER_BATCH_BUCKETS: Incomplete
FLASHINFER_MAX_SEQLEN_BUCKETS: Incomplete
FLASHINFER_CUDNN_WORKSPACE_SIZE_BYTES: Incomplete

def add_padding_to_seqlens(
    seq: np.ndarray, batch_size: int, padding_value: int
) -> np.ndarray: ...
def bucket_flashinfer_max_seqlen(real_max_seqlen: int) -> int: ...

class MMEncoderAttention(CustomOp):
    @classmethod
    def compute_max_seqlen(
        cls, attn_backend: AttentionBackendEnum, cu_seqlens: np.ndarray
    ) -> int: ...
    @classmethod
    def maybe_compute_sequence_lengths(
        cls, attn_backend: AttentionBackendEnum, cu_seqlens: np.ndarray
    ) -> np.ndarray | None: ...
    @classmethod
    def maybe_recompute_cu_seqlens(
        cls,
        attn_backend: AttentionBackendEnum,
        cu_seqlens: np.ndarray,
        hidden_size: int,
        tp_size: int,
    ) -> np.ndarray: ...
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    num_kv_heads: Incomplete
    layer_name: Incomplete
    num_queries_per_kv: Incomplete
    attn_backend: Incomplete
    is_flash_attn_backend: Incomplete
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
    ) -> None: ...
    @classmethod
    def enabled(cls) -> bool: ...
    def view_qkv_to_4d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        bsz: int,
        q_len: int,
        kv_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...
    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def forward_cpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,
        sequence_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
