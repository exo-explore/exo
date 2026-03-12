import torch
from _typeshed import Incomplete
from collections.abc import Iterator
from vllm.config import VllmConfig as VllmConfig
from vllm.platforms import current_platform as current_platform
from vllm.v1.attention.backend import AttentionBackend as AttentionBackend
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec as LoadStoreSpec,
    OffloadingManager as OffloadingManager,
)
from vllm.v1.kv_offload.arc_manager import ARCOffloadingManager as ARCOffloadingManager
from vllm.v1.kv_offload.backends.cpu import CPUBackend as CPUBackend
from vllm.v1.kv_offload.lru_manager import LRUOffloadingManager as LRUOffloadingManager
from vllm.v1.kv_offload.mediums import (
    CPULoadStoreSpec as CPULoadStoreSpec,
    GPULoadStoreSpec as GPULoadStoreSpec,
)
from vllm.v1.kv_offload.reuse_manager import (
    FilterReusedOffloadingManager as FilterReusedOffloadingManager,
)
from vllm.v1.kv_offload.spec import OffloadingSpec as OffloadingSpec
from vllm.v1.kv_offload.worker.cpu_gpu import (
    CpuGpuOffloadingHandlers as CpuGpuOffloadingHandlers,
)
from vllm.v1.kv_offload.worker.worker import OffloadingHandler as OffloadingHandler

class CPUOffloadingSpec(OffloadingSpec):
    num_blocks: Incomplete
    eviction_policy: str
    def __init__(
        self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
    ) -> None: ...
    def get_manager(self) -> OffloadingManager: ...
    def get_handlers(
        self,
        kv_caches: dict[str, torch.Tensor],
        attn_backends: dict[str, type[AttentionBackend]],
    ) -> Iterator[
        tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]
    ]: ...
