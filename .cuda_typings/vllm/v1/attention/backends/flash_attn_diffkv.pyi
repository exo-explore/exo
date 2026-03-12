import torch
from .flash_attn import (
    FlashAttentionBackend as FlashAttentionBackend,
    FlashAttentionImpl as FlashAttentionImpl,
    FlashAttentionMetadata as FlashAttentionMetadata,
    cascade_attention as cascade_attention,
)
from _typeshed import Incomplete
from vllm.logger import init_logger as init_logger
from vllm.v1.attention.backend import AttentionType as AttentionType
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func as flash_attn_varlen_func,
    is_flash_attn_varlen_func_available as is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import get_kv_cache_layout as get_kv_cache_layout
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv as triton_reshape_and_cache_flash_diffkv,
)

logger: Incomplete

class FlashAttentionDiffKVBackend(FlashAttentionBackend):
    head_size_v: int
    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]: ...

class FlashAttentionDiffKVImpl(FlashAttentionImpl):
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
