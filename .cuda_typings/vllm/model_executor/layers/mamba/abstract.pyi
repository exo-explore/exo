import abc
import torch
from abc import abstractmethod
from collections.abc import Iterable
from vllm.config import VllmConfig as VllmConfig
from vllm.model_executor.layers.attention_layer_base import (
    AttentionLayerBase as AttentionLayerBase,
)
from vllm.v1.attention.backend import AttentionBackend as AttentionBackend
from vllm.v1.attention.selector import get_mamba_attn_backend as get_mamba_attn_backend
from vllm.v1.kv_cache_interface import (
    KVCacheSpec as KVCacheSpec,
    MambaSpec as MambaSpec,
)

class MambaBase(AttentionLayerBase, metaclass=abc.ABCMeta):
    kv_cache: tuple[torch.Tensor, ...]
    @abstractmethod
    def get_state_shape(self) -> Iterable[tuple[int, ...]]: ...
    @property
    @abstractmethod
    def mamba_type(self) -> str: ...
    @abstractmethod
    def get_state_dtype(self) -> tuple[torch.dtype, ...]: ...
    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None: ...
    def get_attn_backend(self) -> type[AttentionBackend]: ...
