from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import (
    KVConnectorBaseType as KVConnectorBaseType,
)
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory as KVConnectorFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1 as KVConnectorBase_V1,
    KVConnectorRole as KVConnectorRole,
)
from vllm.v1.kv_cache_interface import KVCacheConfig as KVCacheConfig

def get_kv_transfer_group() -> KVConnectorBaseType: ...
def has_kv_transfer_group() -> bool: ...
def is_v1_kv_transfer_group(connector: KVConnectorBaseType | None = None) -> bool: ...
def ensure_kv_transfer_initialized(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig | None = None
) -> None: ...
def ensure_kv_transfer_shutdown() -> None: ...
