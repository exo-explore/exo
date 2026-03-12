import torch
from _typeshed import Incomplete
from dataclasses import dataclass
from typing_extensions import Self
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.utils.math_utils import cdiv as cdiv
from vllm.utils.torch_utils import get_dtype_size as get_dtype_size

logger: Incomplete

@dataclass(frozen=True)
class KVCacheSpec:
    block_size: int
    @property
    def page_size_bytes(self) -> int: ...
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...
    def copy_with_new_block_size(self, block_size: int) -> Self: ...
    @classmethod
    def merge(cls, specs: list[Self]) -> Self: ...

@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    page_size_padded: int | None = ...
    @property
    def page_size_bytes(self) -> int: ...
    @property
    def real_page_size_bytes(self) -> int: ...

@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    head_size_v: int = ...
    sliding_window: int | None = ...
    attention_chunk_size: int | None = ...
    def __post_init__(self) -> None: ...
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...
    @classmethod
    def merge_window_sizes(cls, window_sizes: set[int]) -> int | None: ...
    @classmethod
    def merge(cls, specs: list[Self]) -> Self: ...
    @property
    def real_page_size_bytes(self) -> int: ...

@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    cache_dtype_str: str | None = ...
    @property
    def real_page_size_bytes(self) -> int: ...
    @classmethod
    def merge(cls, specs: list[Self]) -> Self: ...

@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    attention_chunk_size: int
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...

@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...

@dataclass(frozen=True)
class MambaSpec(KVCacheSpec):
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype]
    page_size_padded: int | None = ...
    mamba_type: str = ...
    mamba_cache_mode: str = ...
    num_speculative_blocks: int = ...
    @property
    def page_size_bytes(self) -> int: ...
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...

@dataclass(frozen=True)
class EncoderOnlyAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...

@dataclass(frozen=True)
class CrossAttentionSpec(AttentionSpec):
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...

@dataclass(frozen=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    sink_len: int | None = ...
    @classmethod
    def merge(cls, specs: list[Self]) -> Self: ...

@dataclass(frozen=True)
class UniformTypeKVCacheSpecs(KVCacheSpec):
    kv_cache_specs: dict[str, KVCacheSpec]
    @property
    def page_size_bytes(self) -> int: ...
    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int: ...
    @classmethod
    def is_uniform_type(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> bool: ...
    @classmethod
    def from_specs(cls, kv_cache_specs: dict[str, KVCacheSpec]) -> Self | None: ...

@dataclass
class KVCacheTensor:
    size: int
    shared_by: list[str]

@dataclass
class KVCacheGroupSpec:
    layer_names: list[str]
    kv_cache_spec: KVCacheSpec

@dataclass
class KVCacheConfig:
    num_blocks: int
    kv_cache_tensors: list[KVCacheTensor]
    kv_cache_groups: list[KVCacheGroupSpec]
    @property
    def has_mamba_layers(self) -> bool: ...
    @property
    def needs_kv_cache_zeroing(self) -> bool: ...
