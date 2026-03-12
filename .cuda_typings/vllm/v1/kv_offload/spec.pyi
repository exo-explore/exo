import abc
import torch
from _typeshed import Incomplete
from abc import ABC, abstractmethod
from collections.abc import Iterator
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.v1.attention.backend import AttentionBackend as AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec as LoadStoreSpec,
    OffloadingManager as OffloadingManager,
)
from vllm.v1.kv_offload.worker.worker import OffloadingHandler as OffloadingHandler

logger: Incomplete

class OffloadingSpec(ABC, metaclass=abc.ABCMeta):
    vllm_config: Incomplete
    kv_cache_config: Incomplete
    extra_config: Incomplete
    gpu_block_size: Incomplete
    offloaded_block_size: Incomplete
    def __init__(
        self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None
    ) -> None: ...
    @abstractmethod
    def get_manager(self) -> OffloadingManager: ...
    @abstractmethod
    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]: ...
