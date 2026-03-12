import torch
from _typeshed import Incomplete
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey as QuantKey,
    kFp8StaticTensorSym as kFp8StaticTensorSym,
)
from vllm.v1.attention.backend import (
    AttentionLayer as AttentionLayer,
    AttentionType as AttentionType,
    MultipleOf as MultipleOf,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionMetadata as FlashAttentionMetadata,
)
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend as RocmAttentionBackend,
    RocmAttentionImpl as RocmAttentionImpl,
    RocmAttentionMetadataBuilder as RocmAttentionMetadataBuilder,
)

logger: Incomplete

class RocmAiterUnifiedAttentionBackend(RocmAttentionBackend):
    accept_output_buffer: bool
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool: ...
    @classmethod
    def supports_head_size(cls, head_size: int) -> bool: ...
    @classmethod
    def supports_mm_prefix(cls) -> bool: ...
    @classmethod
    def supports_sink(cls) -> bool: ...
    forward_includes_kv_cache_update: bool
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["RocmAiterUnifiedAttentionImpl"]: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool: ...
    @staticmethod
    def get_builder_cls() -> type["RocmAttentionMetadataBuilder"]: ...
    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool: ...

class RocmAiterUnifiedAttentionImpl(RocmAttentionImpl):
    def fused_output_quant_supported(self, quant_key: QuantKey): ...
    unified_attention: Incomplete
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = ...,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None: ...
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
    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ): ...
    def fused_rope_kvcache_supported(self): ...
    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ): ...
