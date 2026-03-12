from _typeshed import Incomplete
from collections.abc import Callable as Callable
from enum import Enum, EnumMeta
from vllm.logger import init_logger as init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname
from vllm.v1.attention.backend import AttentionBackend as AttentionBackend

logger: Incomplete

class _AttentionBackendEnumMeta(EnumMeta):
    def __getitem__(cls, name: str): ...

class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASH_ATTN_DIFFKV = (
        "vllm.v1.attention.backends.flash_attn_diffkv.FlashAttentionDiffKVBackend"
    )
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    ROCM_ATTN = "vllm.v1.attention.backends.rocm_attn.RocmAttentionBackend"
    ROCM_AITER_MLA = "vllm.v1.attention.backends.mla.rocm_aiter_mla.AiterMLABackend"
    ROCM_AITER_TRITON_MLA = (
        "vllm.v1.attention.backends.mla.aiter_triton_mla.AiterTritonMLABackend"
    )
    ROCM_AITER_FA = (
        "vllm.v1.attention.backends.rocm_aiter_fa.AiterFlashAttentionBackend"
    )
    ROCM_AITER_MLA_SPARSE = (
        "vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse.ROCMAiterMLASparseBackend"
    )
    TORCH_SDPA = ""
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    FLASHINFER_MLA = (
        "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
    )
    FLASHINFER_MLA_SPARSE = "vllm.v1.attention.backends.mla.flashinfer_mla_sparse.FlashInferMLASparseBackend"
    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    CUTLASS_MLA = "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend"
    FLASHMLA = "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
    FLASHMLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
    )
    FLASH_ATTN_MLA = "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"
    NO_ATTENTION = "vllm.v1.attention.backends.no_attention.NoAttentionBackend"
    FLEX_ATTENTION = "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"
    TREE_ATTN = "vllm.v1.attention.backends.tree_attn.TreeAttentionBackend"
    ROCM_AITER_UNIFIED_ATTN = "vllm.v1.attention.backends.rocm_aiter_unified_attn.RocmAiterUnifiedAttentionBackend"
    CPU_ATTN = "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
    CUSTOM = None
    def get_path(self, include_classname: bool = True) -> str: ...
    def get_class(self) -> type[AttentionBackend]: ...
    def is_overridden(self) -> bool: ...
    def clear_override(self) -> None: ...

class MambaAttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    MAMBA1 = "vllm.v1.attention.backends.mamba1_attn.Mamba1AttentionBackend"
    MAMBA2 = "vllm.v1.attention.backends.mamba2_attn.Mamba2AttentionBackend"
    SHORT_CONV = "vllm.v1.attention.backends.short_conv_attn.ShortConvAttentionBackend"
    LINEAR = "vllm.v1.attention.backends.linear_attn.LinearAttentionBackend"
    GDN_ATTN = "vllm.v1.attention.backends.gdn_attn.GDNAttentionBackend"
    CUSTOM = None
    def get_path(self, include_classname: bool = True) -> str: ...
    def get_class(self) -> type[AttentionBackend]: ...
    def is_overridden(self) -> bool: ...
    def clear_override(self) -> None: ...

MAMBA_TYPE_TO_BACKEND_MAP: Incomplete

def register_backend(
    backend: AttentionBackendEnum | MambaAttentionBackendEnum,
    class_path: str | None = None,
    is_mamba: bool = False,
) -> Callable[[type], type]: ...
