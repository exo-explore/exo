import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing import ClassVar
from vllm._aiter_ops import rocm_aiter_ops as rocm_aiter_ops
from vllm.config import VllmConfig as VllmConfig
from vllm.config.cache import CacheDType as CacheDType
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    get_mla_dims as get_mla_dims,
)
from vllm.model_executor.models.deepseek_v2 import Indexer as Indexer
from vllm.triton_utils import tl as tl, triton as triton
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionCGSupport as AttentionCGSupport,
    AttentionLayer as AttentionLayer,
    AttentionMetadata as AttentionMetadata,
    AttentionMetadataBuilder as AttentionMetadataBuilder,
    CommonAttentionMetadata as CommonAttentionMetadata,
    MultipleOf as MultipleOf,
    SparseMLAAttentionImpl as SparseMLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index as triton_convert_req_index_to_global_index,
)
from vllm.v1.kv_cache_interface import AttentionSpec as AttentionSpec

logger: Incomplete

@triton.jit
def fetch_id_to_ragged_kernel(
    in_tensor_ptr,
    cumsum_ptr,
    out_tensor_ptr,
    in_tensor_ptr_stride,
    TOPK: tl.constexpr,
    TOKEN_NUM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
): ...
def fetch_id_to_ragged_triton(
    in_tensor: torch.Tensor, cumsum: torch.Tensor, out_tensor: torch.Tensor, topk
): ...

class ROCMAiterMLASparseBackend(AttentionBackend):
    accept_output_buffer: bool
    supported_dtypes: ClassVar[list[torch.dtype]]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]]
    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_metadata_cls() -> type["ROCMAiterMLASparseMetadata"]: ...
    @staticmethod
    def get_builder_cls() -> type["ROCMAiterMLASparseMetadataBuilder"]: ...
    @staticmethod
    def get_impl_cls() -> type["ROCMAiterMLASparseImpl"]: ...
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]: ...
    @classmethod
    def is_mla(cls) -> bool: ...
    @classmethod
    def is_sparse(cls) -> bool: ...

@dataclass
class ROCMAiterMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int
    num_actual_tokens: int
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    qo_indptr: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indptr_rest: torch.Tensor
    block_size: int = ...
    topk_tokens: int = ...

@dataclass
class ROCMAiterMLASparseMetadataBuilder(
    AttentionMetadataBuilder[ROCMAiterMLASparseMetadata]
):
    kv_cache_spec = ...
    model_config = ...
    device = ...
    num_heads = ...
    mla_dims = ...
    topk_tokens = ...
    topk_tokens_tensor = ...
    max_model_len_tensor = ...
    dummy_block_table = ...
    req_id_per_token_buffer = ...
    qo_indptr = ...
    paged_kv_last_page_len = ...
    paged_kv_indices = ...
    paged_kv_indptr = ...
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None: ...
    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> ROCMAiterMLASparseMetadata: ...

def reference_mla_sparse_prefill(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float, d_v: int
) -> tuple[torch.Tensor, torch.Tensor]: ...

class ROCMAiterMLASparseImpl(SparseMLAAttentionImpl[ROCMAiterMLASparseMetadata]):
    num_heads: Incomplete
    head_size: Incomplete
    scale: Incomplete
    num_kv_heads: Incomplete
    kv_cache_dtype: Incomplete
    kv_lora_rank: int
    softmax_scale: Incomplete
    topk_indices_buffer: torch.Tensor | None
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        topk_indice_buffer: torch.Tensor | None = None,
        indexer: Indexer | None = None,
        **mla_args,
    ) -> None: ...
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...
