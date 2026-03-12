from vllm.config import VllmConfig as VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase as ECConnectorBase,
    ECConnectorRole as ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import (
    ECConnectorFactory as ECConnectorFactory,
)

def get_ec_transfer() -> ECConnectorBase: ...
def has_ec_transfer() -> bool: ...
def ensure_ec_transfer_initialized(vllm_config: VllmConfig) -> None: ...
