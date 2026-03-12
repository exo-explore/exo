from _typeshed import Incomplete
from vllm.v1.attention.backends.mla.rocm_aiter_mla import (
    AiterMLABackend as AiterMLABackend,
    AiterMLAImpl as AiterMLAImpl,
)

class AiterTritonMLABackend(AiterMLABackend):
    @staticmethod
    def get_name() -> str: ...
    @staticmethod
    def get_impl_cls() -> type["AiterTritonMLAImpl"]: ...

class AiterTritonMLAImpl(AiterMLAImpl):
    flash_attn_varlen_func: Incomplete
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
        **mla_args,
    ) -> None: ...
