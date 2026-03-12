from _typeshed import Incomplete
from collections.abc import Callable as Callable
from vllm.config import VllmConfig as VllmConfig
from vllm.config.kv_transfer import KVTransferConfig as KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBase as KVConnectorBase,
    KVConnectorBaseType as KVConnectorBaseType,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorRole as KVConnectorRole,
    supports_hma as supports_hma,
)
from vllm.logger import init_logger as init_logger
from vllm.utils.func_utils import supports_kw as supports_kw
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig

logger: Incomplete

class KVConnectorFactory:
    @classmethod
    def register_connector(
        cls, name: str, module_path: str, class_name: str
    ) -> None: ...
    @classmethod
    def create_connector(
        cls,
        config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> KVConnectorBase: ...
    @classmethod
    def get_connector_class_by_name(
        cls, connector_name: str
    ) -> type[KVConnectorBaseType]: ...
    @classmethod
    def get_connector_class(
        cls, kv_transfer_config: KVTransferConfig
    ) -> type[KVConnectorBaseType]: ...
