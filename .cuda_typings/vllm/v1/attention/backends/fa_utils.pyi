from _typeshed import Incomplete
from flash_attn import flash_attn_varlen_func as flash_attn_varlen_func
from vllm._custom_ops import reshape_and_cache_flash as reshape_and_cache_flash
from vllm._xpu_ops import xpu_ops as xpu_ops
from vllm.logger import init_logger as init_logger
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant as vllm_is_batch_invariant,
)
from vllm.platforms import current_platform as current_platform
from vllm.vllm_flash_attn import get_scheduler_metadata as get_scheduler_metadata

logger: Incomplete

def get_flash_attn_version(
    requires_alibi: bool = False, head_size: int | None = None
) -> int | None: ...
def flash_attn_supports_fp8() -> bool: ...
def flash_attn_supports_sinks() -> bool: ...
def flash_attn_supports_mla(): ...
def is_flash_attn_varlen_func_available() -> bool: ...
