from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.config import VllmConfig as VllmConfig
from vllm.logger import init_logger as init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig
from vllm.v1.kv_offload.spec import OffloadingSpec as OffloadingSpec

logger: Incomplete

class OffloadingSpecFactory:
    @classmethod
    def register_spec(cls, name: str, module_path: str, class_name: str) -> None: ...
    @classmethod
    def create_spec(
        cls, config: VllmConfig, kv_cache_config: KVCacheConfig | None
    ) -> OffloadingSpec: ...
