import torch
from _typeshed import Incomplete
from typing import NamedTuple
from vllm.config.cache import CacheDType as CacheDType
from vllm.logger import init_logger as init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname as resolve_obj_by_qualname
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionType as AttentionType,
)
from vllm.v1.attention.backends.registry import (
    MAMBA_TYPE_TO_BACKEND_MAP as MAMBA_TYPE_TO_BACKEND_MAP,
    MambaAttentionBackendEnum as MambaAttentionBackendEnum,
)

logger: Incomplete

class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = ...
    has_sink: bool = ...
    use_sparse: bool = ...
    use_mm_prefix: bool = ...
    use_per_head_quant_scales: bool = ...
    attn_type: str = ...

def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
) -> type[AttentionBackend]: ...
def get_mamba_attn_backend(mamba_type: str) -> type[AttentionBackend]: ...
