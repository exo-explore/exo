import abc
from abc import ABC, abstractmethod
from vllm.config import VllmConfig as VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend as AttentionBackend,
    AttentionImpl as AttentionImpl,
)
from vllm.v1.kv_cache_interface import KVCacheSpec as KVCacheSpec

class AttentionLayerBase(ABC, metaclass=abc.ABCMeta):
    impl: AttentionImpl
    @abstractmethod
    def get_attn_backend(self) -> type[AttentionBackend]: ...
    @abstractmethod
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None: ...
